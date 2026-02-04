# 🧬 Improving 2D Diffusion Models for 3D Medical Imaging with Inter-Slice Consistent Stochasticity (ISCS)

[![License: Apache-2.0](https://img.shields.io/badge/License-Apache--2.0-blue.svg)](LICENSE)
[![Framework](https://img.shields.io/badge/Framework-PyTorch-red.svg)](https://pytorch.org/)
[![OpenReview](https://img.shields.io/badge/Paper-OpenReview-8A2BE2.svg)](https://openreview.net/forum?id=R5ETdN6ifA)
[![arXiv](https://img.shields.io/badge/Paper-arXiv-B31B1B.svg)](#)

Official PyTorch implementation for the ICLR 2026 paper:
**_Improving 2D Diffusion Models for 3D Medical Imaging with Inter-Slice Consistent Stochasticity_**.

This repository provides demo pipelines for:

- 🩻 Cone-beam CT (CBCT) reconstruction under limited-angle CT (LACT) and sparse-view CT (SVCT)
- 🧠 MRI z-axis super-resolution (ZSR) / slice interpolation

## 🧾 Overview

3D medical imaging is crucial for clinical diagnosis and scientific research, but learning **3D** diffusion priors is often difficult due to limited data availability and heavy training costs. A common compromise is to train diffusion models on **2D** slices and stack them for 3D inverse problems—but the intrinsic randomness in diffusion sampling can cause severe **inter-slice discontinuities**.

We introduce **Inter-Slice Consistent Stochasticity (ISCS)**, a simple yet effective strategy to improve 3D coherence by **controlling the consistency of stochastic noise components during sampling**. This aligns sampling trajectories across slices **without adding new loss terms, optimization steps, or extra computational cost**. ISCS is plug-and-play and can be dropped into existing 2D-trained diffusion-based 3D reconstruction pipelines, yielding improved performance across several medical imaging tasks. ✨

![SVCT-30](./figures/SVCT-30-256.gif)

Figure: qualitative comparison on SVCT (30 views).

---

## 🚀 Quick start

1) Prepare a 3D volume (NIfTI: `.nii` / `.nii.gz`) and a diffusion checkpoint.
2) Edit paths in the demo scripts:
   - `recon_CBCT.sh` for LACT/SVCT CBCT
   - `recon_ZSR.sh` for MRI ZSR
3) Run:

```bash
# LACT / SVCT (CBCT)
bash recon_CBCT.sh

# MRI z-axis super-resolution (ZSR)
bash recon_ZSR.sh
```

## 🛠️ Installation

This project is research code and assumes a CUDA-capable environment.

### 📦 Dependencies

At minimum, you will need the following Python packages:

- `torch`, `torchvision`, `numpy`, `PyYAML`, `ml-collections`
- `SimpleITK` (I/O for NIfTI volumes)
- [carterbox-torch-radon](https://github.com/carterbox/torch-radon) (CT forward/back-projection operators)
- `astra-toolbox` (used in `recon_CBCT.py` to generate FDK and iterative reconstruction baselines)

## 🗂️ Data format

Both demos take a 3D volume as input:

- File format: NIfTI (`.nii` / `.nii.gz`)
- Axis convention: the code reads the volume with SimpleITK and processes it as a tensor of shape `[1, D, H, W]` (CBCT) or a coupled representation for ZSR

Notes:

- CBCT demo expects CT-like intensities; it clamps to a HU range and normalizes to `[0, 1]` internally.
- ZSR demo normalizes by the volume min/max and then runs z-axis super-resolution at the specified factor.

## ▶️ Running the demos

### 🩻 CBCT: LACT and SVCT

Use `recon_CBCT.sh` as a template and set:

- `DATA`: path to your input NIfTI volume
- `CHECKPOINT_PATH`: path to a pretrained diffusion checkpoint (`.pth`)
- `CONFIG_PATH`: model config YAML (e.g., `configs/ve/AAPM_256_ncsnpp_Chung.yaml`)

The CBCT demo *simulates* projections from the input volume using the forward operator and then reconstructs from those measurements. You do not need a separate sinogram file.

Key arguments:

- `--task`: `LACT` (limited-angle) or `SVCT` (sparse-view)
  - `LACT`: `--degree` is the angular coverage in degrees (e.g., 90)
  - `SVCT`: `--degree` is the number of views (e.g., 20)
- `--slice-begin/--slice-end`: reconstruct a slice range (if both are `0`, the full volume is used)
- `--recon-size`: in-plane resize for reconstruction (default: 256)

### 🧠 MRI: Z-axis super-resolution (ZSR)

Use `recon_ZSR.sh` as a template and set:

- `DATA`: path to your input NIfTI volume
- `CHECKPOINT_PATH`: path to a pretrained diffusion checkpoint (`.pth`)
- `CONFIG_PATH`: model config YAML (e.g., `configs/ve/BMR_ZSR_256.yaml`)

Key arguments:

- `--degree`: z-axis super-resolution factor (e.g., 5)
- `--slice-begin/--slice-end`: optional sub-volume selection (if both are `0`, the full volume is used)

## 📤 Outputs

Each run creates a timestamped folder under `results/` and saves:

- Inputs / measurements (e.g., `measurement.nii.gz` or `y.nii.gz`)
- Baseline reconstructions (e.g., FDK / iterative reconstruction for CBCT, pseudo-inverse for ZSR)
- Ground truth / input volume (as processed by the pipeline)
- Final reconstruction (`recon.nii` or `.nii.gz`)
- Quantitative metrics (PSNR/SSIM; additional slice-wise metrics printed to stdout)

## 📝 Notes

- 🗺️ Paths in `recon_CBCT.sh` / `recon_ZSR.sh` are placeholders; update them before running.

## 🧱 Repository structure

High-level layout:

- `recon_CBCT.py`, `recon_CBCT.sh`: CBCT demo entrypoints
- `recon_MRI_ZSR.py`, `recon_ZSR.sh`: MRI ZSR demo entrypoints
- `configs/`: model configs (VE-SDE / NCSN++)
- `models/`: score model definitions
- `algorithms/`: reconstruction algorithms (DDS/DDNM variants, TV/ADMM utilities)
- `physics/`: measurement operators (CT, ZSR)
- `op/`: custom CUDA ops (compiled on first import)
- `utils/`: data I/O, checkpoint loading, metrics, CLI args

## 📖 Citation

If you use this code in your research, please cite:

```bibtex
@inproceedings{du2026ISCS,
  title     = {Improving 2D Diffusion Models for 3D Medical Imaging with Inter-Slice Consistent Stochasticity},
  author    = {Du, Chenhe and Wu, Qing and Tian, Xuanyu and Yu, Jingyi and Wei, Hongjiang and Zhang, Yuyao},
  booktitle = {The Fourteenth International Conference on Learning Representations},
  year      = {2026},
  url       = {https://openreview.net/forum?id=R5ETdN6ifA},
}
```

## 🔐 License

This project is licensed under the Apache License 2.0. See `LICENSE` for details.

## 🙏 Acknowledgements

This repository is built upon [score_sde_pytorch](https://github.com/yang-song/score_sde_pytorch). We thank the authors for releasing their code.
