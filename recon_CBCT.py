import datetime
import os
from pathlib import Path

import numpy as np
import SimpleITK as sitk
import torch
import torch_radon
import yaml
from ml_collections import config_dict
from torchvision.transforms import functional as TF

import models.ncsnpp
import utils.args
import utils.data
import utils.model
import utils.result
from algorithms import ADMM_TV
from algorithms.CBCT import DDNM, DDS
from physics.ct import CBCT_carterbox

torch.set_num_threads(20)
HU_MAX = 1600
HU_MIN = -1000


def get_orientation_code(img: sitk.Image) -> str:
    f = sitk.DICOMOrientImageFilter()
    return f.GetOrientationFromDirectionCosines(img.GetDirection())


def load_and_preprocess_image(args, config):
    gt_sitk = sitk.ReadImage(args.data)
    ori_orient = get_orientation_code(gt_sitk)
    gt_sitk = sitk.DICOMOrient(gt_sitk, "LPI")

    metainfo_spacing = gt_sitk.GetSpacing()
    metainfo_direction = gt_sitk.GetDirection()

    # [D,H,W] -> numpy view
    gt_image = sitk.GetArrayViewFromImage(gt_sitk)

    D = gt_image.shape[0]
    if args.slice_begin == 0 and args.slice_end == 0:
        args.slice_begin, args.slice_end = 0, D
    args.slice_begin = max(0, args.slice_begin)
    args.slice_end = min(D, args.slice_end)

    gt_image = gt_image[args.slice_begin : args.slice_end]  # [d,H,W]
    gt_image = torch.tensor(gt_image, dtype=torch.float32, device=config.device)
    gt_image = gt_image.unsqueeze(0)  # [1,d,H,W]

    gt_image = utils.data.HU_to_norm_01(gt_image, HU_MAX, HU_MIN)

    if gt_image.shape[-1] != args.recon_size:
        metainfo_spacing = (
            metainfo_spacing[0] / (args.recon_size / gt_image.shape[-1]),
            metainfo_spacing[1] / (args.recon_size / gt_image.shape[-2]),
            metainfo_spacing[2],
        )

        gt_image = TF.resize(
            gt_image,
            size=[args.recon_size, args.recon_size],
            interpolation=TF.InterpolationMode.BILINEAR,
            antialias=True,
        )

    metainfo = {}
    metainfo["spacing"] = metainfo_spacing
    metainfo["direction"] = metainfo_direction

    return gt_image, metainfo


def get_view_indices(task, degree, view_full_num):
    if task == "LACT":
        view_limited_num = int(view_full_num * (degree / 360))
        view_limited_idx = np.linspace(0, view_limited_num, view_limited_num, endpoint=False, dtype=int)
    elif task == "SVCT":
        view_limited_num = degree
        view_limited_idx = np.linspace(0, view_full_num, view_limited_num, endpoint=False, dtype=int)
    else:
        raise ValueError(f"Unsupported task: {task}")

    return view_limited_idx


def add_noise_if_needed(measurement, sino_noise):
    if sino_noise > 0:
        level = np.sqrt(sino_noise) if sino_noise < 100 else sino_noise
        noise_type = "gaussian" if sino_noise < 100 else "poisson"
        measurement, SNR = utils.data.add_sino_noise_guassian(measurement, level, noise_type)
        print(f"Add {noise_type} Noise to Measurement with level: {level} and SNR: {SNR}")
    return measurement


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
    view_full_num = 360
    view_limited_idx = get_view_indices(args.task, args.degree, view_full_num)
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

    gt_image, metainfo = load_and_preprocess_image(args, config)  # [0,1]
    print(f"图像尺寸: {gt_image.shape}, 间距: {metainfo['spacing']}")
    D, H, W = gt_image.shape[1], gt_image.shape[2], gt_image.shape[3]

    angles_FV = np.linspace(0, 2 * np.pi, view_full_num, endpoint=False).astype(np.float32)
    angles_LV = angles_FV[view_limited_idx]

    volume = torch_radon.volumes.Volume3D()
    volume.set_size(gt_image.shape[1], gt_image.shape[2], gt_image.shape[3])

    measure_model = CBCT_carterbox(
        angles_FV=angles_FV,
        angle_LV=angles_LV,
        det_count_u=512,
        det_count_v=512,
        det_spacing_u=2,
        det_spacing_v=2,
        src_dist=700,
        det_dist=500,
        volume=volume,
    )

    print(gt_image.shape)
    projections = measure_model.A_FV(gt_image).float().to(config.device).clone().detach()
    measurement = measure_model.A(gt_image).float().to(config.device).clone().detach()
    measurement = torch.tensor(measurement.cpu().numpy(), dtype=torch.float32, device=config.device)

    print(f"Projections shape: {projections.shape}, Measurement shape: {measurement.shape}")

    measurement = add_noise_if_needed(measurement, args.sino_noise)
    import astra

    gt_np = gt_image.squeeze().cpu().numpy()
    gt_np = np.array(gt_np, dtype=np.float32)

    vol_geom = astra.create_vol_geom(gt_np.shape[1], gt_np.shape[2], gt_np.shape[0])

    proj_geom_LV = astra.create_proj_geom("cone", 2.0, 2.0, 512, 512, angles_LV, 700, 500)
    proj_geom_FV = astra.create_proj_geom("cone", 2.0, 2.0, 512, 512, angles_FV, 700, 500)

    astra_fdk_lv = utils.data.fdk_reconstruct(gt_np, proj_geom_LV, vol_geom)
    astra_fdk_fv = utils.data.fdk_reconstruct(gt_np, proj_geom_FV, vol_geom)
    astra_cgls_lv = utils.data.astra_IR(gt_np, proj_geom_LV, vol_geom, recon_algo="SIRT3D_CUDA", iter=100)

    fbp_lv = astra_fdk_lv
    fbp_fv = astra_fdk_fv
    cgls_lv = astra_cgls_lv

    fbp_lv = torch.tensor(fbp_lv, dtype=torch.float32, device=config.device).unsqueeze(0)
    fbp_fv = torch.tensor(fbp_fv, dtype=torch.float32, device=config.device).unsqueeze(0)
    cgls_lv = torch.tensor(cgls_lv, dtype=torch.float32, device=config.device).unsqueeze(0)

    # Create Save Path
    # ----------------------------------
    save_root = Path(
        f"results/{config_name}/{data_name}/CBCT/{args.task}-{args.degree}/{args.method}/{problem}/{datetime.datetime.now():%y%m%d_%H%M%S}/"
    )
    save_root.mkdir(parents=True, exist_ok=True)

    with open(save_root / "args.yaml", "w") as f:
        yaml.dump(vars(args), f)

    utils.result.save_nii_image(measurement, os.path.join(save_root, "measurement.nii.gz"))
    utils.result.save_nii_image(fbp_lv, os.path.join(save_root, "FBP-LV.nii.gz"), sitk_info=metainfo)
    utils.result.save_nii_image(fbp_fv, os.path.join(save_root, "FBP-FV.nii.gz"), sitk_info=metainfo)
    utils.result.save_nii_image(gt_image, os.path.join(save_root, "GT.nii.gz"), sitk_info=metainfo)
    utils.result.save_nii_image(cgls_lv, os.path.join(save_root, "astra-IR.nii.gz"), sitk_info=metainfo)

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
                x_init=fbp_lv.squeeze().unsqueeze(1),
                y=measurement,
                cg_iter=args.num_cg,
                w_dz=args.w_dz,
                w_tik=args.w_tik,
                renoise_method=args.renoise_method,
                save_path=save_root,
                noise_control=args.noise_control,
                use_init=args.use_init,
                save_intermediates=True,
            )
        elif args.method == "DDNM":
            print("Run DDNM!")
            recon_pipeline = DDNM.DDNM(sde, score_model, config=config, measure_model=measure_model)
            x = recon_pipeline.reconstruct(
                x_init=cgls_lv.squeeze().unsqueeze(1),
                y=measurement,
                renoise_method=args.renoise_method,
                save_path=save_root,
                noise_control=args.noise_control,
                use_init=args.use_init,
                save_intermediates=True,
                vol_geom=vol_geom,
                proj_geom_LV=proj_geom_LV,
            )
        elif args.method == "uncondition":
            print("Run un-condition generation!")
            x = torch.tensor(np.random.uniform(0, 1, size=gt_image.shape), dtype=torch.float32, device=config.device)
    elif args.method == "ADMM-TV":
        print("Run ADMM-TV generation!")
        x_init = torch.zeros_like(fbp_lv)
        recon_pipeline = ADMM_TV.ADMM_TV(
            measure_model=measure_model, img_shape=x_init.shape, lamb=3, rho=50, outer_iter=30, inner_iter=20
        )
        x = recon_pipeline.reconstruct(x_init=x_init, y=measurement)
    else:
        raise ValueError(f"Invalid method: {args.method}. only support {diffusion_methods + ['ADMM-TV']}.")

    utils.result.save_nii_image(x, os.path.join(save_root, "recon.nii.gz"), sitk_info=metainfo)
    print(f"Save to: {save_root}")

    # * ---------------------------------------
    # * FBP and Metrics
    fbp_lv = fbp_lv.view(D, 1, H, W)
    cgls_lv = cgls_lv.view(D, 1, H, W)
    gt_image = gt_image.view(D, 1, H, W)
    x = x.view(D, 1, H, W)

    data_range_gt = (gt_image.max() - gt_image.min()).item()

    psnr, ssim = utils.result.cal_metrics(fbp_lv, gt_image, save_root / "FBP-LV_metrics")
    print("--------------------------------")
    print(f"FBP-LV PSNR: {psnr:.4f}\nFBP-LV SSIM: {ssim:.4f}")

    metrics = utils.result.compute_slice_metrics_optimized(fbp_lv, gt_image, data_range=data_range_gt)
    print(f"{'View':<10}{'PSNR':>12}{'SSIM':>12}")
    print(f"{'Axial':<10}{metrics['axial']['PSNR_mean']:12.4f}{metrics['axial']['SSIM_mean']:12.4f}")
    print(f"{'Coronal':<10}{metrics['coronal']['PSNR_mean']:12.4f}{metrics['coronal']['SSIM_mean']:12.4f}")
    print(f"{'Sagittal':<10}{metrics['sagittal']['PSNR_mean']:12.4f}{metrics['sagittal']['SSIM_mean']:12.4f}")

    # * ---------------------------------------
    psnr, ssim = utils.result.cal_metrics(x, gt_image, save_root / "recon_metrics", sitk_info=metainfo)
    print("--------------------------------")
    print(f"recon PSNR: {psnr:.4f}\nrecon SSIM: {ssim:.4f}")

    metrics = utils.result.compute_slice_metrics_optimized(x, gt_image, data_range=data_range_gt)
    print(f"{'View':<10}{'PSNR':>12}{'SSIM':>12}")
    print(f"{'Axial':<10}{metrics['axial']['PSNR_mean']:12.4f}{metrics['axial']['SSIM_mean']:12.4f}")
    print(f"{'Coronal':<10}{metrics['coronal']['PSNR_mean']:12.4f}{metrics['coronal']['SSIM_mean']:12.4f}")
    print(f"{'Sagittal':<10}{metrics['sagittal']['PSNR_mean']:12.4f}{metrics['sagittal']['SSIM_mean']:12.4f}")


if __name__ == "__main__":
    args = utils.args.build_parser()
    main(args)
