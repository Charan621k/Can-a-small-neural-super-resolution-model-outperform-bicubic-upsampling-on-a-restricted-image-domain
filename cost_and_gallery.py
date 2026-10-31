#!/usr/bin/env python3
"""
Cost and upscaling-gallery numbers for the Topic 29 paper. Nothing is retrained.

Usage (same folder as train_sr.py, after the full run):
    python cost_and_gallery.py --root data --models full

Writes into <models>/:
  upscaling.pdf / upscaling.png   rows = 4 test images chosen by RANK of per-image gain (not by taste),
                                  columns = LR input, bicubic, network, original, with per-image PSNR
  cost.csv, cost_rows.tex          inference time (CPU, and GPU if present) and training time per size
Prints everything it writes, so you can copy the numbers into the paper.
"""
import argparse
import csv
import glob
import os
import platform
import re

import numpy as np
import torch

import train_sr as T


def cpu_name():
    try:
        with open("/proc/cpuinfo") as f:
            for line in f:
                if line.startswith("model name"):
                    return line.split(":", 1)[1].strip()
    except OSError:
        pass
    return platform.processor() or platform.machine()


def psnr1(a, b):
    mse = float(((a - b) ** 2).mean())
    return 10 * np.log10(1.0 / max(mse, 1e-10))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="data")
    ap.add_argument("--models", default="full")
    ap.add_argument("--size", type=int, default=1020, help="training size of the model used for the gallery")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--indices", default="", help="optional: comma-separated test-image indices to show instead of the rank-based choice")
    args = ap.parse_args()

    cpu = torch.device("cpu")
    path = os.path.join(args.models, f"model_n{args.size}_s{args.seed}.pt")
    net = T.ESPCN()
    net.load_state_dict(torch.load(path, map_location=cpu))
    net.eval()

    te_hr = T.to_tensor(T.load_split(args.root, "test", args.smoke))
    te_lr = T.make_lr(te_hr)
    print(f"test images: {len(te_hr)}; model: {path}")

    # ---------------- upscaling gallery -----------------
    with torch.no_grad():
        lr_f = te_lr.float() / 255
        bic = torch.cat([T.quantize8(T.bicubic_up(lr_f[i:i + 256])) for i in range(0, len(lr_f), 256)])
        out = torch.cat([T.quantize8(net(lr_f[i:i + 256])) for i in range(0, len(lr_f), 256)])
    hr = te_hr.float() / 255
    p_b = np.array([psnr1(bic[i], hr[i]) for i in range(len(hr))])
    p_n = np.array([psnr1(out[i], hr[i]) for i in range(len(hr))])
    gain = p_n - p_b
    order = np.argsort(gain)
    picks = [int(order[int(q * (len(order) - 1))]) for q in (0.05, 0.35, 0.65, 0.95)]
    if args.indices:
        picks = [int(v) for v in args.indices.split(",")]
    print("gallery images (index, gain dB):", [(i, round(float(gain[i]), 2)) for i in picks])

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(len(picks), 4, figsize=(6.2, 1.6 * len(picks) + 0.2))
    heads = ["LR input (32x32)", "Bicubic", "Network", "Original"]
    for r, i in enumerate(picks):
        lr_big = torch.nn.functional.interpolate(lr_f[i:i + 1], size=(T.HR_SIZE, T.HR_SIZE), mode="nearest")[0]
        panels = [(lr_big, ""), (bic[i], f"{p_b[i]:.2f} dB"), (out[i], f"{p_n[i]:.2f} dB"), (hr[i], "")]
        for c, (im, lab) in enumerate(panels):
            a = ax[r, c]
            a.imshow(im.permute(1, 2, 0).numpy(), interpolation="nearest")
            a.set_xticks([]); a.set_yticks([])
            if r == 0:
                a.set_title(heads[c], fontsize=8)
            if lab:
                a.set_xlabel(lab, fontsize=7, labelpad=1)
        ax[r, 2].set_ylabel(f"gain {gain[i]:+.2f} dB", fontsize=7, rotation=0, labelpad=2, ha="left", va="center")
        ax[r, 2].yaxis.set_label_position("right")
    fig.tight_layout(pad=0.3)
    fig.savefig(os.path.join(args.models, "upscaling.pdf"))
    fig.savefig(os.path.join(args.models, "upscaling.png"), dpi=200)

    # ---------------- inference time -----------------
    rows = []
    devs = [("CPU", cpu, cpu_name())]
    if torch.cuda.is_available():
        devs.append(("GPU", torch.device("cuda"), torch.cuda.get_device_name(0)))
    print(f"\nthreads (CPU): {torch.get_num_threads()}; torch {torch.__version__}")
    for label, dev, name in devs:
        m = T.ESPCN(); m.load_state_dict(torch.load(path, map_location=dev)); m.to(dev).eval()
        reps = [T.benchmark(m, te_lr, dev) for _ in range(3)]  # 3 independent measurements
        b = np.array([r["bicubic"] for r in reps]); n = np.array([r["network"] for r in reps])
        rows.append(dict(device=label, name=name, bicubic_ms=b.mean(), bicubic_sd=b.std(ddof=1),
                         network_ms=n.mean(), network_sd=n.std(ddof=1), ratio=n.mean() / b.mean()))
        print(f"{label} ({name}): bicubic {b.mean():.3f} ms (sd {b.std(ddof=1):.3f}), network {n.mean():.3f} ms "
              f"(sd {n.std(ddof=1):.3f}), ratio {n.mean() / b.mean():.1f}x, "
              f"{1000 / b.mean():,.0f} vs {1000 / n.mean():,.0f} images/s")

    # ---------------- training time -----------------
    print("\ntraining time per run (from the saved run files):")
    print(f"{'size':>6}{'runs':>6}{'mean s':>9}{'min s':>8}{'max s':>8}{'best step (mean)':>18}")
    trows = []
    for n_ in sorted({int(re.search(r'run_n(\d+)_', f).group(1)) for f in glob.glob(os.path.join(args.models, "run_n*_s*.npz"))}):
        rr = [np.load(f) for f in sorted(glob.glob(os.path.join(args.models, f"run_n{n_}_s*.npz")))]
        secs = np.array([float(r["train_secs"]) for r in rr]); bs = np.array([float(r["best_step"]) for r in rr])
        trows.append(dict(train_n=n_, runs=len(rr), mean_s=secs.mean(), min_s=secs.min(), max_s=secs.max(), best_step_mean=bs.mean(),
                          best_steps=" ".join(str(int(x)) for x in bs)))
        print(f"{n_:>6}{len(rr):>6}{secs.mean():>9.0f}{secs.min():>8.0f}{secs.max():>8.0f}{bs.mean():>18.0f}   steps: {trows[-1]['best_steps']}")
    if not trows:
        print("  no run_n*_s*.npz files found in", args.models)

    with open(os.path.join(args.models, "cost.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
        if trows:
            f.write("\n")
            w2 = csv.DictWriter(f, fieldnames=list(trows[0].keys())); w2.writeheader(); w2.writerows(trows)
    with open(os.path.join(args.models, "cost_rows.tex"), "w") as f:
        for r in rows:
            f.write(f"{r['device']} & {r['bicubic_ms']:.3f} & {r['network_ms']:.3f} & {r['ratio']:.1f} \\\\\n")
    print(f"\nSaved cost.csv, cost_rows.tex, upscaling.pdf/png in {args.models}/")
    print("Paste the CPU name into the paper; the table rows are in cost_rows.tex.")


if __name__ == "__main__":
    main()
