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
