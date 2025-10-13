# Improving 2D Diffusion Models for 3D Medical Imaging with Inter‑Slice Consistent Stochasticity 🚀

Official PyTorch implementation for "Improving 2D Diffusion Models for 3D Medical Imaging with Inter‑Slice Consistent Stochasticity"

## 🖼️ Visualization

![SVCT-30](./figures/merge.gif)
Fig. 1: Qualitative results of DDS and DDS+ISCS on a representative sample for SVCT of 30 views.

## 🧪 Running Reconstructions

We provide demo scripts for two inverse problems.

Each experiments can be run by simply running

```bash
# For LACT or SVCT
bash recon_CBCT.sh

# For MRI isotropic SR
bash recon_ZSR.sh
```

## 📜 Citation

If you find our work interesting, please consider citing:

```bibtex
@article{YourCitation2025,
  title   = {Improving 2D Diffusion Models for 3D Medical Imaging with Inter‑Slice Consistent Stochasticity},
  author  = {First Author and Second Author and Others},
  journal = {Journal / Conference},
  year    = {2025}
}
```

## 🔐 License

See the `LICENSE` file for details.

---
