#!/usr/bin/env python3
"""
Robustness check for Topic 29.

Question: the networks were trained on low-res images made by antialiased bicubic downsampling,
the same kernel family as the baseline. Do they STILL beat bicubic upsampling when the low-res
input is made in a different way? If the gain vanishes, the network mostly learned to undo one
specific operation.

Nothing is retrained. The saved models from train_sr.py are evaluated on the 6,149 test images
with five different downsampling methods. The baseline is always bicubic upsampling of THE SAME
low-res input, and the decision rule is the same as before (mean gain >= 0.5 dB, 95% CI excludes 0).

Usage (same folder as train_sr.py, after the full run):
    python eval_shift.py --root data --models full

Writes <models>/shift_summary.csv and <models>/shift.png.
The first kernel ("bicubic_aa") is the training kernel: its numbers must match full/summary.csv.
If they do not, something is wrong; do not trust the other rows.
"""
import argparse
import csv
import glob
import os
import re

import numpy as np
import torch
import torch.nn.functional as F

import train_sr as T

LR = (T.LR_SIZE, T.LR_SIZE)


def down_gauss(x, sigma=1.5):
    """Optical-blur style: Gaussian blur (sigma 1.5 HR pixels), then sample at the LR pixel centres."""
    k = torch.arange(9, dtype=torch.float32) - 4
    g = torch.exp(-k ** 2 / (2 * sigma ** 2))
    g = g / g.sum()
    w = (g[:, None] * g[None, :])[None, None].repeat(3, 1, 1, 1)
    xb = F.conv2d(F.pad(x, (4, 4, 4, 4), mode="reflect"), w, groups=3)
    return F.interpolate(xb, size=LR, mode="bilinear", align_corners=False)


KERNELS = {
    "bicubic_aa": lambda x: F.interpolate(x, size=LR, mode="bicubic", align_corners=False, antialias=True),  # training kernel (control)
    "area": lambda x: F.interpolate(x, size=LR, mode="area"),
    "bilinear_aa": lambda x: F.interpolate(x, size=LR, mode="bilinear", align_corners=False, antialias=True),
    "gauss_blur": down_gauss,
    "bicubic_no_aa": lambda x: F.interpolate(x, size=LR, mode="bicubic", align_corners=False, antialias=False),  # aliased
}


def make_lr(fn, hr_u8):
    out = []
    for i in range(0, len(hr_u8), 256):
        x = hr_u8[i:i + 256].float() / 255
        out.append((fn(x).clamp(0, 1) * 255).round().to(torch.uint8))
    return torch.cat(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="data", help="same folder as in train_sr.py (cached test images)")
    ap.add_argument("--models", default="full", help="folder with model_n{size}_s{seed}.pt from train_sr.py")
    ap.add_argument("--threshold", type=float, default=0.5)
    ap.add_argument("--residual", action="store_true", help="only if the models were trained with --residual")
    ap.add_argument("--smoke", action="store_true", help="fake-data test of the code")
    args = ap.parse_args()

    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    files = sorted(glob.glob(os.path.join(args.models, "model_n*_s*.pt")))
    if not files:
        raise SystemExit(f"no model_n*_s*.pt files in {args.models}")
    models = {}
    for f in files:
        m = re.search(r"model_n(\d+)_s(\d+)\.pt$", f)
        net = T.ESPCN().to(dev)
        net.load_state_dict(torch.load(f, map_location=dev))
        net.eval()
        models[(int(m.group(1)), int(m.group(2)))] = net
    sizes = sorted({n for n, _ in models})
    print(f"device: {dev.type}; {len(models)} models; sizes {sizes}")

    te_hr = T.to_tensor(T.load_split(args.root, "test", args.smoke))
    print(f"test images: {len(te_hr)}")

    rows = []
    print(f"\n{'kernel':<15}{'train n':>8}{'bicubic':>9}{'net':>9}{'mean dPSNR':>12}{'95% CI':>18}{'seed sd':>9}  verdict")
    for kname, fn in KERNELS.items():
        lr = make_lr(fn, te_hr)
        bic, _ = T.evaluate(None, lr, te_hr, dev, want_ssim=False)
        for n in sizes:
            seeds = sorted(s for (nn, s) in models if nn == n)
            net = np.stack([T.evaluate(models[(n, s)], lr, te_hr, dev, residual=args.residual, want_ssim=False)[0] for s in seeds])
            d_img = (net - bic[None]).mean(axis=0)
            lo, hi = T.boot_ci(d_img)
            sd = (net - bic[None]).mean(axis=1).std(ddof=1) if len(seeds) > 1 else float("nan")
            v = T.verdict(d_img.mean(), lo, hi, args.threshold)
            rows.append(dict(kernel=kname, train_n=n, seeds=len(seeds), psnr_bicubic=bic.mean(), psnr_net=net.mean(),
                             d_psnr=d_img.mean(), ci_lo=lo, ci_hi=hi, seed_sd=sd, verdict=v))
            print(f"{kname:<15}{n:>8}{bic.mean():>9.3f}{net.mean():>9.3f}{d_img.mean():>12.3f}"
                  f"  [{lo:>6.3f},{hi:>6.3f}]{sd:>9.3f}  {v}")

    path = os.path.join(args.models, "shift_summary.csv")
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(7.5, 4.2))
    names = list(KERNELS)
    for j, n in enumerate(sizes):
        ys = [next(r["d_psnr"] for r in rows if r["kernel"] == k and r["train_n"] == n) for k in names]
        xs = np.arange(len(names)) + (j - (len(sizes) - 1) / 2) * 0.14
        ax.plot(xs, ys, "o", label=f"{n} train images")
    ax.axhline(args.threshold, color="tab:green", label=f"{args.threshold} dB rule")
    ax.axhline(0, color="gray", ls="--", lw=1)
    ax.set_xticks(np.arange(len(names)))
    ax.set_xticklabels(names, fontsize=8)
    ax.set_ylabel("mean gain over bicubic (dB)")
    ax.set_title("Gain when the low-res input is made differently (same trained models)", fontsize=10)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(args.models, "shift.png"), dpi=200)

    bad = sorted({r["kernel"] for r in rows if r["verdict"] != "PASSES the rule"})
    print(f"\nSaved: {path} and shift.png")
    print("CHECK 1: the 'bicubic_aa' rows must match summary.csv from the training run.")
    if bad:
        print("CHECK 2: NOT passing the rule for: " + ", ".join(bad) + " -> the gain depends on how the low-res image is made. Say so in the paper.")
    else:
        print("CHECK 2: the rule passes for every kernel -> the gain does not depend on the bicubic downsampling kernel (for these five).")


if __name__ == "__main__":
    main()
