import datetime
import os
from pathlib import Path

import SimpleITK as sitk
import torch
import yaml
from ml_collections import config_dict
from torchvision import transforms

import models.ncsnpp
import utils.args
import utils.data
import utils.model
import utils.result
from algorithms.MRI_ZSR import DDNM, DDS
from physics import zsr

torch.set_num_threads(20)
HU_MAX = 440
HU_MIN = 20


def get_orientation_code(img: sitk.Image) -> str:
    # Obtain the direction string (e.g., 'RAI', 'LPI', etc.);
    # note that the syntax varies slightly across different versions of SimpleITK.
    f = sitk.DICOMOrientImageFilter()
    return f.GetOrientationFromDirectionCosines(img.GetDirection())


def load_and_preprocess_image(args, config):
    gt_sitk = sitk.ReadImage(args.data)
    gt_image = sitk.GetArrayViewFromImage(gt_sitk)

    D = gt_image.shape[0]
    if args.slice_begin == 0 and args.slice_end == 0:
        args.slice_begin, args.slice_end = 0, D
    args.slice_begin = max(0, args.slice_begin)
    args.slice_end = min(D, args.slice_end)

    gt_image = gt_image[args.slice_begin : args.slice_end]  # [d,H,W]
    gt_image = torch.tensor(gt_image, dtype=torch.float32, device=config.device).unsqueeze(1)
    gt_image = gt_image.permute(2, 1, 3, 0)

    gt_image = utils.data.HU_to_norm_01(gt_image, gt_image.max().item(), gt_image.min().item())

    if gt_image.shape[-1] != args.recon_size:
        gt_image = transforms.Resize(
            args.recon_size, interpolation=transforms.InterpolationMode.BILINEAR, antialias=True
        )(gt_image)
    return gt_image


def main(args):
    # Load and Setting Configs
    # ----------------------------------
    config_path = args.config_path
    config_name = config_path.split("/")[-1].replace(".yaml", "")
    with open(config_path, "r") as f:
        config_data = yaml.safe_load(f)

    config = config_dict.ConfigDict(config_data)
    config.device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")

    data_path = Path(args.data)
    data_name = data_path.stem

    # Build problem string and include weight fields only when they are non-zero
    parts = [
        f"{args.task}",
        f"{args.degree}",
        f"{args.slice_begin:04d}",
        f"{args.slice_end:04d}",
        f"{args.NFE}",
        f"{args.use_init}",
    ]
    # Only include regularization weights when they are non-zero to keep names concise
    if getattr(args, "num_cg", 0):
        parts.append(f"nCG-{args.num_cg}")
    if getattr(args, "w_dps", 0):
        parts.append(f"wDPS-{args.w_dps}")
    if getattr(args, "w_tik", 0):
        parts.append(f"wTIK-{args.w_tik}")
    if getattr(args, "w_dz", 0):
        parts.append(f"wDZ-{args.w_dz}")

    # Always include noise_control and renoise_method (may be None)
    parts.append(str(args.noise_control))
    parts.append(str(args.renoise_method))

    problem = "_".join(map(str, parts))

    print(f"Task: {problem}")

    # Data
    gt_image = load_and_preprocess_image(args, config)  # [0,1]

    # * ---------------------------------------
    # * Setup Measurement Model A
    measure_model = zsr.ZAxisSuperResolution(args.degree)
    y = measure_model.A(gt_image).float().to(config.device).clone().detach()
    y = torch.tensor(y.cpu().numpy()).to(config.device)

    ATy = measure_model.A_dagger(y).clone().detach()
    ATy = measure_model.couple(measure_model.decouple(ATy))

    # Create Save Path
    # ----------------------------------
    save_root = Path(
        f"results/{config_name}/{data_name}/{args.task}-{args.degree}/{args.method}/{problem}/{datetime.datetime.now():%y%m%d_%H%M%S}/"
    )
    save_root.mkdir(parents=True, exist_ok=True)

    utils.result.save_nii_image(y, os.path.join(save_root, "y.nii.gz"))
    utils.result.save_nii_image(ATy, os.path.join(save_root, "ATy.nii.gz"))
    utils.result.save_nii_image(gt_image, os.path.join(save_root, "GT.nii.gz"))

    diffusion_methods = ["DDS", "DDNM", "uncondition"]
    if args.method in diffusion_methods:
        # Setup Diffusion Model
        # ----------------------------------
        ckpt_filename = args.checkpoint_path
        from sde_lib import VESDE

        sde = VESDE(sigma_min=args.sigma_min, sigma_max=args.sigma_max, N=args.NFE)
        score_model = utils.model.get_score_models(config, ckpt_filename)

        # Create Pipeline
        # ----------------------------------
        if args.method == "DDS":
            print("Run DDS!")
            recon_pipeline = DDS.DDS(sde, score_model, config=config, measure_model=measure_model)
            x = recon_pipeline.reconstruct(
                x_init=ATy,
                y=y,
                cg_iter=args.num_cg,
                w_dz=args.w_dz,
                save_path=save_root,
                use_init=args.use_init,
                save_intermediates=True,
                noise_control=args.noise_control,
                renoise_method=args.renoise_method,
            )
        elif args.method == "DDNM":
            print("Run DDNM!")
            recon_pipeline = DDNM.DDNMReconstructor(
                sde, score_model, config=config, measure_model=measure_model, factor=args.degree
            )
            x = recon_pipeline.reconstruct(
                x_init=ATy,
                y=y,
                save_path=save_root,
                use_init=args.use_init,
                save_intermediates=True,
                noise_control=args.noise_control,
            )
        elif args.method == "uncondition":
            print("Run un-condition generation!")
    else:
        raise ValueError("Invalid method")
    # * ---------------------------------------

    data_range_gt = (gt_image.max() - gt_image.min()).item()

    # Save Recon Image
    # ----------------------------------
    utils.result.save_nii_image(x, os.path.join(save_root, "recon.nii"))

    psnr, ssim = utils.result.cal_metrics(x, gt_image, save_root)

    print(f"Save to: {save_root}")
    print(f"PSNR: {psnr}\nSSIM: {ssim}")

    metrics = utils.result.compute_slice_metrics_optimized(x, gt_image, data_range=data_range_gt)
    print(f"{'View':<10}{'PSNR':>12}{'SSIM':>12}")
    print(f"{'Axial':<10}{metrics['axial']['PSNR_mean']:12.4f}{metrics['axial']['SSIM_mean']:12.4f}")
    print(f"{'Coronal':<10}{metrics['coronal']['PSNR_mean']:12.4f}{metrics['coronal']['SSIM_mean']:12.4f}")
    print(f"{'Sagittal':<10}{metrics['sagittal']['PSNR_mean']:12.4f}{metrics['sagittal']['SSIM_mean']:12.4f}")


if __name__ == "__main__":
    args = utils.args.build_parser()
    main(args)
