# LD3D Refactor Drop-in

This folder contains refactored versions of the provided training (`3_*.py`) and inference (`4_*.py`) scripts.

## What changed
- **Single GPU by default** (no `torch.distributed.launch`, no `mp.spawn`).
- Optional DDP via `torchrun`.
- Faster dataloading: `pin_memory`, `persistent_workers`, `prefetch_factor`, and a robust `dict_collate`.
- **TensorBoard fixed**: step + epoch scalars are written and `flush()` is called.
- **Early stopping**, periodic checkpoints (`latest`, `best`, and `epochXXXX`).
- Validation each epoch with a tqdm **progress bar** (stdout). Logs go to timestamped log files.
- Metrics:
  - **3D**: MAE, PSNR (averaged over modalities)
  - **2D slice-wise (added)**: SSIM, PSNR, MAE (averaged over slices, configurable stride)
  - Optional LPIPS proxy (MONAI PerceptualLoss in fake-3D mode) in the supervised UNet script.

## Commands
Supervised UNet baseline:
```bash
python 3_train_ours.py --config configs/config_unet3d_rcg.yaml
```

RCG (needs pretrained RDM ckpt):
```bash
python 3_train_rcg_optimized_metrics_final.py --config configs/config_unet3d_rcg.yaml --rdm-ckpt /path/to/autoencoder.pt
```

Inference:
```bash
python 4_inference_ours.py --config configs/config_unet3d_rcg.yaml --ckpt /path/to/ckpt_best.pt
```


## Commands (train / inference)

### General notes
- **Single GPU default:** use `python ...`
- **Multi-GPU optional:** use `torchrun --standalone --nproc_per_node=2 ...`
- **Fast smoke test:** add `--max-train-batches 10 --max-val-batches 10` (runs only 10 batches per epoch / validation)

### Training

**OURS (UNet raw 3D)**
```bash
python 3_train_ours.py --config configs/config_unet3d_rcg.yaml
python 3_train_ours.py --config configs/config_unet3d_rcg.yaml --max-train-batches 10 --max-val-batches 10
torchrun --standalone --nproc_per_node=2 3_train_ours.py --config configs/config_unet3d_rcg.yaml
```

**UNEST**
```bash
python 3_train_unest.py --config configs/config_unest.yaml
python 3_train_unest.py --config configs/config_unest.yaml --max-train-batches 10 --max-val-batches 10
torchrun --standalone --nproc_per_node=2 3_train_unest.py --config configs/config_unest.yaml
```

**MedSyn**
```bash
python 3_train_medsyn.py --config configs/config_medsyn.yaml
python 3_train_medsyn.py --config configs/config_medsyn.yaml --max-train-batches 10 --max-val-batches 10
torchrun --standalone --nproc_per_node=2 3_train_medsyn.py --config configs/config_medsyn.yaml
```

**I2I-Mamba**
```bash
python 3_train_i2imamba.py --config configs/config_i2imamba.yaml
python 3_train_i2imamba.py --config configs/config_i2imamba.yaml --max-train-batches 10 --max-val-batches 10
torchrun --standalone --nproc_per_node=2 3_train_i2imamba.py --config configs/config_i2imamba.yaml
```

**Vanilla diffusion**
```bash
python 3_train_vanilla_diffusion.py --config configs/config_unet3d_rcg.yaml
python 3_train_vanilla_diffusion.py --config configs/config_unet3d_rcg.yaml --max-train-batches 10 --max-val-batches 10
torchrun --standalone --nproc_per_node=2 3_train_vanilla_diffusion.py --config configs/config_unet3d_rcg.yaml
```

**RCG (requires pretrained RDM checkpoint)**
```bash
python 3_train_rcg_optimized_metrics_final.py --config configs/config_unet3d_rcg.yaml --rdm-ckpt /path/to/autoencoder.pt
python 3_train_rcg_optimized_metrics_final.py --config configs/config_unet3d_rcg.yaml --rdm-ckpt /path/to/autoencoder.pt --max-train-batches 10 --max-val-batches 10
torchrun --standalone --nproc_per_node=2 3_train_rcg_optimized_metrics_final.py --config configs/config_unet3d_rcg.yaml --rdm-ckpt /path/to/autoencoder.pt
```

### Inference (metrics per dataset)

All inference scripts expect `--ckpt` pointing to your trained checkpoint.
```bash
python 4_inference_ours.py --config configs/config_unet3d_rcg.yaml --ckpt /path/to/ckpt_best.pt
python 4_inference_unest.py --config configs/config_unest.yaml --ckpt /path/to/ckpt_best.pt
python 4_inference_medsyn.py --config configs/config_medsyn.yaml --ckpt /path/to/ckpt_best.pt
python 4_inference_i2imamba.py --config configs/config_i2imamba.yaml --ckpt /path/to/ckpt_best.pt
python 4_inference_cyclegan.py --config configs/config_cyclegan.yaml --ckpt /path/to/ckpt_best.pt
```

### NIfTI integrity scan (corruption check)
```bash
python -m utils.check_nifti_integrity --max 0 --workers 16
```
