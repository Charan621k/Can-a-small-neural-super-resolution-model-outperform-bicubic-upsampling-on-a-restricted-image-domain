"""
Topic 29: can a small neural super-resolution model beat bicubic upsampling?
Oxford Flowers 102, scale x4, ESPCN-style network (37,200 parameters).

What this script does (matches the exposé):
  1. Loads the OFFICIAL Flowers 102 splits (train 1,020 / val 1,020 / test 6,149),
     resizes the shorter side to 128 and centre-crops to 128x128 (= ground truth, HR).
  2. Makes the 32x32 low-res input (LR) with antialiased bicubic downsampling x4.
  3. Baseline: bicubic upsampling x4.
  4. Trains the network for several training-set sizes and seeds. The ONLY thing that
     changes between runs is the number of training images (same number of steps).
  5. Evaluates on the test set: per-image PSNR and SSIM, paired difference to bicubic,
     bootstrap 95% CI, the 0.5 dB decision rule, timing, FLOPs, crops, failure plot.

Usage:
  python train_sr.py                      # full experiment (sizes 100 250 500 1020, seeds 0 1 2)
  python train_sr.py --sizes 1020 --seeds 0 --steps 3000     # quick first test
  python train_sr.py --analyze-only       # redo tables/plots from saved runs
  python train_sr.py --smoke              # tiny fake-data test of the code (no download)
"""
import argparse
import copy
import csv
import os
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

SCALE = 4
HR_SIZE = 128
LR_SIZE = HR_SIZE // SCALE


# ----------------------------------------------------------------------------------
# data
# ----------------------------------------------------------------------------------
def fake_split(n, seed):
    """Random images with edges, only for --smoke (checks the code, says nothing about results)."""
    from PIL import Image, ImageDraw
    rng = np.random.default_rng(seed)
    out = []
    for _ in range(n):
        base = (rng.random((8, 8, 3)) * 255).astype(np.uint8)
        im = Image.fromarray(base).resize((HR_SIZE, HR_SIZE), Image.BICUBIC)
        d = ImageDraw.Draw(im)
        for _ in range(6):
            x0, y0 = [int(v) for v in rng.integers(0, 100, 2)]
            w, h = [int(v) for v in rng.integers(8, 28, 2)]
            d.rectangle([x0, y0, x0 + w, y0 + h], fill=tuple(int(v) for v in rng.integers(0, 255, 3)))
        out.append(np.asarray(im))
    return np.stack(out)


def load_split(root, split, smoke=False):
    """Returns uint8 array N x 128 x 128 x 3 (cached as .npy after the first call)."""
    if smoke:
        n = {"train": 60, "val": 20, "test": 30}[split]
        return fake_split(n, {"train": 1, "val": 2, "test": 3}[split])
    cache = os.path.join(root, f"flowers102_{split}_{HR_SIZE}.npy")
    if os.path.exists(cache):
        return np.load(cache)
    from PIL import Image
    from torchvision.datasets import Flowers102
    ds = Flowers102(root, split=split, download=True)
    arr = []
    for img, _ in ds:
        img = img.convert("RGB")
        w, h = img.size
        s = HR_SIZE / min(w, h)  # shorter side -> 128, then centre crop
        img = img.resize((max(HR_SIZE, round(w * s)), max(HR_SIZE, round(h * s))), Image.BICUBIC)
        w, h = img.size
        l, t = (w - HR_SIZE) // 2, (h - HR_SIZE) // 2
        arr.append(np.asarray(img.crop((l, t, l + HR_SIZE, t + HR_SIZE)), dtype=np.uint8))
    arr = np.stack(arr)
    np.save(cache, arr)
    return arr


def to_tensor(a):
    return torch.from_numpy(a).permute(0, 3, 1, 2).contiguous()  # uint8, N x 3 x H x W


def make_lr(hr_u8):
    """HR uint8 -> LR uint8 with antialiased bicubic downsampling (same call for train and test)."""
    out = []
    for i in range(0, len(hr_u8), 256):
        x = hr_u8[i:i + 256].float() / 255
        lr = F.interpolate(x, size=(LR_SIZE, LR_SIZE), mode="bicubic", align_corners=False, antialias=True)
        out.append((lr.clamp(0, 1) * 255).round().to(torch.uint8))
    return torch.cat(out)


def bicubic_up(lr):
    """Baseline. NOTE: PyTorch's bicubic uses a = -0.75; PIL/Matlab use -0.5. Say which one you used."""
    return F.interpolate(lr, size=(HR_SIZE, HR_SIZE), mode="bicubic", align_corners=False).clamp(0, 1)


def quantize8(x):
    return (x.clamp(0, 1) * 255).round() / 255  # outputs are scored as 8-bit images, like saved files


# ----------------------------------------------------------------------------------
# model and metrics
# ----------------------------------------------------------------------------------
class ESPCN(nn.Module):
    def __init__(self, scale=SCALE, ch=3):
        super().__init__()
        self.body = nn.Sequential(
            nn.Conv2d(ch, 64, 5, padding=2), nn.ReLU(inplace=True),
            nn.Conv2d(64, 32, 3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(32, ch * scale ** 2, 3, padding=1),
        )
        self.shuffle = nn.PixelShuffle(scale)

    def forward(self, x):
        return self.shuffle(self.body(x))


def n_params(m):
    return sum(p.numel() for p in m.parameters())


def macs_per_image():
    convs = [(3, 64, 5), (64, 32, 3), (32, 48, 3)]
    return sum(ci * co * k * k * LR_SIZE * LR_SIZE for ci, co, k in convs)


def psnr(a, b):  # a, b in [0,1], shape B x 3 x H x W -> B values
    mse = ((a - b) ** 2).mean(dim=(1, 2, 3)).clamp_min(1e-10)
    return 10 * torch.log10(1.0 / mse)


def ssim(a, b):  # standard SSIM: 11x11 Gaussian window, sigma 1.5, per channel, averaged
    g = torch.arange(11, dtype=torch.float32, device=a.device) - 5
    g = torch.exp(-g ** 2 / (2 * 1.5 ** 2))
    g = g / g.sum()
    w = (g[:, None] * g[None, :])[None, None].repeat(3, 1, 1, 1)
    f = lambda x: F.conv2d(x, w, groups=3)
    c1, c2 = 0.01 ** 2, 0.03 ** 2
    ma, mb = f(a), f(b)
    saa, sbb, sab = f(a * a) - ma ** 2, f(b * b) - mb ** 2, f(a * b) - ma * mb
    s = ((2 * ma * mb + c1) * (2 * sab + c2)) / ((ma ** 2 + mb ** 2 + c1) * (saa + sbb + c2))
    return s.mean(dim=(1, 2, 3))


@torch.no_grad()
def predict(model, lr, residual):
    p = bicubic_up(lr) if model is None else model(lr) + (bicubic_up(lr) if residual else 0)
    return quantize8(p)


@torch.no_grad()
def evaluate(model, lr_u8, hr_u8, dev, residual=False, want_ssim=True, bs=128):
    """Per-image PSNR (and SSIM). model=None means the bicubic baseline."""
    if model is not None:
        model.eval()
    ps, ss = [], []
    for i in range(0, len(lr_u8), bs):
        lr = lr_u8[i:i + bs].to(dev).float() / 255
        hr = hr_u8[i:i + bs].to(dev).float() / 255
        p = predict(model, lr, residual)
        ps.append(psnr(p, hr).cpu())
        if want_ssim:
            ss.append(ssim(p, hr).cpu())
    return torch.cat(ps).numpy(), (torch.cat(ss).numpy() if want_ssim else None)


# ----------------------------------------------------------------------------------
# training
# ----------------------------------------------------------------------------------
def train_one(tr_hr, tr_lr, va_hr, va_lr, args, seed, dev, log):
    torch.manual_seed(seed)
    np.random.seed(seed)
    model = ESPCN().to(dev)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode="max", factor=0.5, patience=3)
    tr_hr, tr_lr = tr_hr.to(dev), tr_lr.to(dev)
    n, c = len(tr_lr), args.crop
    best, best_step, best_state, bad = -1.0, 0, None, 0
    t0 = time.time()
    for step in range(1, args.steps + 1):
        model.train()
        idx = torch.randint(0, n, (args.bs,))
        ys = torch.randint(0, LR_SIZE - c + 1, (args.bs,))
        xs = torch.randint(0, LR_SIZE - c + 1, (args.bs,))
        lr_b = torch.stack([tr_lr[i, :, y:y + c, x:x + c] for i, y, x in zip(idx, ys, xs)]).float() / 255
        hr_b = torch.stack([tr_hr[i, :, SCALE * y:SCALE * (y + c), SCALE * x:SCALE * (x + c)]
                            for i, y, x in zip(idx, ys, xs)]).float() / 255
        if np.random.rand() < 0.5:  # light augmentation: flips / transpose, same for LR and HR
            lr_b, hr_b = lr_b.flip(3), hr_b.flip(3)
        if np.random.rand() < 0.5:
            lr_b, hr_b = lr_b.flip(2), hr_b.flip(2)
        if np.random.rand() < 0.5:
            lr_b, hr_b = lr_b.transpose(2, 3), hr_b.transpose(2, 3)
        pred = model(lr_b) + (bicubic_up_crop(lr_b) if args.residual else 0)
        loss = F.l1_loss(pred, hr_b)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        if step % args.val_every == 0:
            v = evaluate(model, va_lr, va_hr, dev, args.residual, want_ssim=False)[0].mean()
            sched.step(v)
            if v > best + 1e-4:
                best, best_step, bad = v, step, 0
                best_state = copy.deepcopy(model.state_dict())
            else:
                bad += 1
            log(f"    step {step:5d}  loss {loss.item():.4f}  val PSNR {v:.3f} dB  (best {best:.3f} @ {best_step})")
            if bad >= args.patience:
                log("    early stop (no validation improvement)")
                break
    model.load_state_dict(best_state if best_state is not None else model.state_dict())
    return model, best, best_step, time.time() - t0


def bicubic_up_crop(lr):  # residual mode on crops: upsample x4 whatever the crop size
    return F.interpolate(lr, scale_factor=SCALE, mode="bicubic", align_corners=False).clamp(0, 1)


# ----------------------------------------------------------------------------------
# statistics, plots, timing
# ----------------------------------------------------------------------------------
def boot_ci(d, B=10000, seed=0):
    """95% bootstrap CI of the mean over images (resample images with replacement)."""
    rng = np.random.default_rng(seed)
    n = len(d)
    means = []
    for _ in range(B // 500):
        idx = rng.integers(0, n, (500, n))
        means.append(d[idx].mean(axis=1))
    means = np.concatenate(means)
    return np.percentile(means, [2.5, 97.5])


def verdict(mean, lo, hi, thr):
    if hi < 0:
        return "worse than bicubic"
    if lo <= 0:
        return "no evidence of a gain"
    return "PASSES the rule" if mean >= thr else "real but below threshold"


def lap_var(hr_u8):
    k = torch.tensor([[0., 1., 0.], [1., -4., 1.], [0., 1., 0.]]).view(1, 1, 3, 3)
    g = hr_u8.float().mean(dim=1, keepdim=True) / 255
    return F.conv2d(g, k).var(dim=(1, 2, 3)).numpy()


def benchmark(model, lr_u8, dev, bs=64, reps=20):
    x = lr_u8[:bs].to(dev).float() / 255
    out = {}
    model.eval()
    for name, fn in (("bicubic", lambda: bicubic_up(x)), ("network", lambda: model(x))):
        with torch.no_grad():
            for _ in range(3):
                fn()
            if dev.type == "cuda":
                torch.cuda.synchronize()
            t0 = time.perf_counter()
            for _ in range(reps):
                fn()
            if dev.type == "cuda":
                torch.cuda.synchronize()
        out[name] = (time.perf_counter() - t0) / reps / len(x) * 1000
    return out


def analyze(args, sizes, seeds, dev, te_hr, te_lr):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rows = []
    print("\n=== RESULTS (test set, %d images) ===" % len(te_hr))
    print(f"{'train n':>8} {'bicubic':>9} {'net':>9} {'mean dPSNR':>11} {'95% CI':>17} "
          f"{'seed sd':>8} {'dSSIM':>8}  verdict")
    for n in sizes:
        runs = [np.load(os.path.join(args.out, f"run_n{n}_s{s}.npz")) for s in seeds
                if os.path.exists(os.path.join(args.out, f"run_n{n}_s{s}.npz"))]
        if not runs:
            continue
        bic = runs[0]["psnr_bic"]
        net = np.stack([r["psnr_net"] for r in runs])  # seeds x images
        d_img = (net - bic[None]).mean(axis=0)  # per-image gain averaged over seeds
        lo, hi = boot_ci(d_img)
        seed_means = (net - bic[None]).mean(axis=1)
        sd = seed_means.std(ddof=1) if len(runs) > 1 else float("nan")
        ds = np.stack([r["ssim_net"] for r in runs]).mean(axis=0) - runs[0]["ssim_bic"]
        v = verdict(d_img.mean(), lo, hi, args.threshold)
        rows.append(dict(train_n=n, seeds=len(runs), psnr_bicubic=bic.mean(), psnr_net=net.mean(),
                         d_psnr=d_img.mean(), ci_lo=lo, ci_hi=hi, seed_sd=sd, d_ssim=ds.mean(), verdict=v))
        print(f"{n:>8} {bic.mean():>9.3f} {net.mean():>9.3f} {d_img.mean():>11.3f} "
              f"[{lo:>6.3f},{hi:>6.3f}] {sd:>8.3f} {ds.mean():>8.4f}  {v}")
    if not rows:
        print("no runs found")
        return
    with open(os.path.join(args.out, "summary.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    # learning curve with the decision threshold
    fig, ax = plt.subplots(figsize=(5.5, 3.6))
    xs = [r["train_n"] for r in rows]
    ax.errorbar(xs, [r["d_psnr"] for r in rows],
                yerr=[[r["d_psnr"] - r["ci_lo"] for r in rows], [r["ci_hi"] - r["d_psnr"] for r in rows]],
                fmt="o-", capsize=4, color="#C2185B")
    ax.axhline(0, color="gray", ls="--", lw=1)
    ax.axhline(args.threshold, color="#2A9D8F", lw=1.5, label=f"{args.threshold} dB rule")
    ax.set_xscale("log")
    ax.set_xticks(xs)
    ax.set_xticklabels([str(x) for x in xs])
    ax.set_xlabel("training images")
    ax.set_ylabel("mean gain over bicubic (dB)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(args.out, "learning_curve.png"), dpi=200)
    plt.close(fig)

    # failure map + crops for the largest size, first seed
    nmax = rows[-1]["train_n"]
    p = os.path.join(args.out, f"run_n{nmax}_s{seeds[0]}.npz")
    mp = os.path.join(args.out, f"model_n{nmax}_s{seeds[0]}.pt")
    if os.path.exists(p) and os.path.exists(mp):
        r = np.load(p)
        d = r["psnr_net"] - r["psnr_bic"]
        lv = lap_var(te_hr)
        fig, ax = plt.subplots(figsize=(5.5, 3.6))
        ax.scatter(lv, d, s=4, alpha=0.4, color="#6A4C93")
        ax.axhline(0, color="gray", ls="--", lw=1)
        ax.set_xscale("log")
        ax.set_xlabel("image detail (variance of Laplacian)")
        ax.set_ylabel("gain over bicubic (dB)")
        fig.tight_layout()
        fig.savefig(os.path.join(args.out, "gain_vs_detail.png"), dpi=200)
        plt.close(fig)

        model = ESPCN().to(dev)
        model.load_state_dict(torch.load(mp, map_location=dev))
        order = np.argsort(d)
        picks = [("worst", order[0]), ("median", order[len(order) // 2]), ("best", order[-1])]
        fig, axes = plt.subplots(3, 4, figsize=(8, 6))
        for row, (name, i) in enumerate(picks):
            lr = te_lr[i:i + 1].to(dev).float() / 255
            hr = te_hr[i].float() / 255
            pb = bicubic_up(lr)[0].cpu()
            pn = predict(model, lr, args.residual)[0].cpu()
            # patch with the most detail (32x32 window, stride 16)
            best_yx, best_v = (0, 0), -1
            for y in range(0, HR_SIZE - 32 + 1, 16):
                for x in range(0, HR_SIZE - 32 + 1, 16):
                    v = hr[:, y:y + 32, x:x + 32].var().item()
                    if v > best_v:
                        best_v, best_yx = v, (y, x)
            y, x = best_yx
            for col, (t, im) in enumerate((("original", hr), ("bicubic", pb), ("network", pn))):
                axes[row, col].imshow(im[:, y:y + 32, x:x + 32].permute(1, 2, 0).numpy(), interpolation="nearest")
                axes[row, col].set_title(f"{name}: {t}" if col == 0 else t, fontsize=8)
                axes[row, col].axis("off")
            err = (pn - hr).abs().mean(0) - (pb - hr).abs().mean(0)  # negative = network better
            axes[row, 3].imshow(err[y:y + 32, x:x + 32].numpy(), cmap="coolwarm", vmin=-0.1, vmax=0.1,
                                interpolation="nearest")
            axes[row, 3].set_title(f"error change ({d[i]:+.2f} dB)", fontsize=8)
            axes[row, 3].axis("off")
        fig.tight_layout()
        fig.savefig(os.path.join(args.out, "crops.png"), dpi=200)
        plt.close(fig)

        t = benchmark(model, te_lr, dev)
        print(f"\nTiming on {dev.type}, batch 64, ms per image: bicubic {t['bicubic']:.3f}, network {t['network']:.3f}")
        print(f"Network: {n_params(model):,} parameters, {macs_per_image() / 1e6:.1f} M MACs per 32x32 input")
    print(f"\nSaved: summary.csv, learning_curve.png, gain_vs_detail.png, crops.png in {args.out}/")


# ----------------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="data")
    ap.add_argument("--out", default="results")
    ap.add_argument("--sizes", type=int, nargs="+", default=[100, 250, 500, 1020])
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--steps", type=int, default=6000, help="max training steps (same for every size)")
    ap.add_argument("--val_every", type=int, default=250)
    ap.add_argument("--patience", type=int, default=8, help="validations without improvement")
    ap.add_argument("--bs", type=int, default=64)
    ap.add_argument("--crop", type=int, default=16, help="LR crop size (HR crop = 4x)")
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--threshold", type=float, default=0.5, help="decision rule in dB")
    ap.add_argument("--residual", action="store_true",
                    help="predict a correction on top of bicubic (changes the model: report it!)")
    ap.add_argument("--analyze-only", action="store_true")
    ap.add_argument("--force", action="store_true", help="retrain even if a run file exists")
    ap.add_argument("--smoke", action="store_true", help="tiny fake-data test of the code")
    args = ap.parse_args()

    os.makedirs(args.root, exist_ok=True)
    os.makedirs(args.out, exist_ok=True)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logf = open(os.path.join(args.out, "log.txt"), "a")

    def log(s):
        print(s, flush=True)
        logf.write(s + "\n")
        logf.flush()

    log(f"device: {dev}, torch {torch.__version__}, params: {n_params(ESPCN()):,}")
    tr_hr = to_tensor(load_split(args.root, "train", args.smoke))
    va_hr = to_tensor(load_split(args.root, "val", args.smoke))
    te_hr = to_tensor(load_split(args.root, "test", args.smoke))
    tr_lr, va_lr, te_lr = make_lr(tr_hr), make_lr(va_hr), make_lr(te_hr)
    log(f"images: train {len(tr_hr)}, val {len(va_hr)}, test {len(te_hr)}")
    sizes = [s for s in args.sizes if s <= len(tr_hr)]

    if not args.analyze_only:
        bic_val = evaluate(None, va_lr, va_hr, dev, want_ssim=False)[0].mean()
        log(f"bicubic val PSNR: {bic_val:.3f} dB  (the network must at least reach this; if it stays far below, "
            f"look for a bug before believing the result)")
        bic_psnr, bic_ssim = evaluate(None, te_lr, te_hr, dev)
        for n in sizes:
            for seed in args.seeds:
                path = os.path.join(args.out, f"run_n{n}_s{seed}.npz")
                if os.path.exists(path) and not args.force:
                    log(f"skip n={n} seed={seed} (exists)")
                    continue
                log(f"== train size {n}, seed {seed}")
                perm = np.random.default_rng(seed).permutation(len(tr_hr))[:n]
                idx = torch.from_numpy(perm)
                model, best, best_step, secs = train_one(tr_hr[idx], tr_lr[idx], va_hr, va_lr, args, seed, dev, log)
                p_net, s_net = evaluate(model, te_lr, te_hr, dev, args.residual)
                np.savez(path, psnr_net=p_net, psnr_bic=bic_psnr, ssim_net=s_net, ssim_bic=bic_ssim,
                         best_val=best, best_step=best_step, train_secs=secs)
                torch.save(model.state_dict(), os.path.join(args.out, f"model_n{n}_s{seed}.pt"))
                log(f"   test PSNR net {p_net.mean():.3f} vs bicubic {bic_psnr.mean():.3f} "
                    f"(mean gain {np.mean(p_net - bic_psnr):+.3f} dB) in {secs:.0f}s")
    analyze(args, sizes, args.seeds, dev, te_hr, te_lr)


if __name__ == "__main__":
    main()
