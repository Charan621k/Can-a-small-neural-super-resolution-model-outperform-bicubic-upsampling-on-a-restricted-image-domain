# Does a small super-resolution network beat bicubic upsampling? (Oxford Flowers 102, x4)

Code for the course paper "Image Processing and Computer Vision", TH Deggendorf (Campus Cham), topic 29.

## Files
- `train_sr.py`: trains the ESPCN-style network (37,200 parameters) and evaluates it against bicubic upsampling.
- `eval_shift.py`: evaluates the saved models on low-resolution input made in five different ways (no retraining).
- `cost_and_gallery.py`: inference and training times, and the upscaling example figure.
- `results/`: `summary.csv`, `shift_summary.csv`, `cost.csv` from the runs reported in the paper.

## Run (PyTorch 2.14.1, torchvision 0.29.1)
```
python train_sr.py --root data --out full          # 4 training sizes x 3 seeds, downloads Flowers 102
python eval_shift.py --root data --models full
python cost_and_gallery.py --root data --models full
```
Add `--smoke` to any script to test it on fake data in a few seconds (says nothing about results).

## Notes
- Decision rule (fixed before the experiments): the network wins if the mean PSNR gain over bicubic is at least 0.5 dB and the 95% bootstrap interval over test images excludes 0.
- Baseline: PyTorch bicubic (Keys kernel, a = -0.75), outputs quantised to 8 bit.
- Dataset: official Oxford Flowers 102 split (1,020 / 1,020 / 6,149), not included; downloaded by torchvision.
- Timings are from one machine (Apple M4 Pro, CPU) and vary between measurement sessions.
