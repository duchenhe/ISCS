# 🧬 Improving 2D Diffusion Models for 3D Medical Imaging with Inter-Slice Consistent Stochasticity (ISCS)

[![OpenReview](https://img.shields.io/badge/Paper-OpenReview-8A2BE2.svg)](https://openreview.net/forum?id=R5ETdN6ifA)
[![arXiv](https://img.shields.io/badge/Paper-arXiv-B31B1B.svg)](https://arxiv.org/abs/2602.04162)

Official PyTorch implementation for the ICLR 2026 paper:
**_Improving 2D Diffusion Models for 3D Medical Imaging with Inter-Slice Consistent Stochasticity_**.

This repository provides demo pipelines for:

- 🩻 Cone-beam CT (CBCT) reconstruction under limited-angle CT (LACT) and sparse-view CT (SVCT)
- 🧠 MRI z-axis super-resolution (ZSR) / slice interpolation

## 🧾 Overview

3D medical imaging is crucial for clinical diagnosis and scientific research, but learning **3D** diffusion priors is often difficult due to limited data availability and heavy training costs. A common compromise is to train diffusion models on **2D** slices and stack them for 3D inverse problems—but the intrinsic randomness in diffusion sampling can cause severe **inter-slice discontinuities**.

> 💡 Intuition: The core idea is straightforward. **If two slices are adjacent in the clean data manifold, they should also reside close to each other in the noise manifold.** Therefore, their stochastic noise structures during the diffusion sampling process should be highly correlated, rather than completely independent.

We introduce **Inter-Slice Consistent Stochasticity (ISCS)**, a simple yet effective strategy to improve 3D coherence by **controlling the consistency of stochastic noise components during sampling**. This aligns sampling trajectories across slices **without adding new loss terms, optimization steps, or extra computational cost**. ISCS is plug-and-play and can be dropped into existing 2D-trained diffusion-based 3D reconstruction pipelines, yielding improved performance across several medical imaging tasks. ✨

![SVCT-30](./figures/SVCT-30-256.gif)

Figure: qualitative comparison on SVCT (30 views).

---

## 🚀 Quick start

1) Prepare a 3D volume (NIfTI: `.nii` / `.nii.gz`) and a diffusion checkpoint. **Pre-trained model weights on CT data can be found in the [GitHub release](https://github.com/duchenhe/ISCS/releases/tag/diffusion-prior) page.**

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

## 🧱 Repository structure

```text
.
├── recon_CBCT.py          # CBCT reconstruction entry point
├── recon_CBCT.sh          # CBCT demo script
├── recon_MRI_ZSR.py       # MRI ZSR reconstruction entry point
├── recon_ZSR.sh           # MRI ZSR demo script
├── configs/ve/            # Model configs
├── models/                # Score network definitions
├── algorithms/            # Reconstruction algorithms
├── physics/               # Measurement operators
├── op/                    # Custom CUDA ops
└── utils/                 # Data I/O, metrics, args, checkpoint loading
```

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

This repository builds upon several fantastic open-source projects. We'd like to express our gratitude to the authors of:

- [score_sde_pytorch](https://github.com/yang-song/score_sde_pytorch)
- [DiffusionMBIR](https://github.com/hyungjin-chung/DiffusionMBIR)
- [DDS](https://github.com/hyungjin-chung/DDS)
- [TPDM](https://github.com/hyn2028/tpdm)