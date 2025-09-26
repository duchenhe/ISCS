import astra
import numpy as np
import torch
import torch.nn.functional as F


def HU_to_norm_01(data, max_val=3200, min_val=-2048):
    data[data < min_val] = min_val
    data[data > max_val] = max_val
    # [0, 1]
    data = (data - min_val) / (max_val - min_val)

    return data


def center_pad_nd(input_tensor, target_shape, value=0.0):
    """
    Center pads the spatial dimensions of a tensor to target_shape.
    Works for NCHW (2D) and NCDHW (3D), etc.

    Args:
        input_tensor: torch.Tensor, shape (N, C, *spatial)
        target_shape: tuple/list, like (H_out, W_out) or (D_out, H_out, W_out)
        value: pad value, default 0

    Returns:
        padded_tensor: torch.Tensor, padded so that original data is centered.

    x = torch.randn(1, 1, 256, 256)
    target_shape = (363, 363)
    y = center_pad_nd(x, target_shape)
    print(y.shape)  # torch.Size([1, 1, 363, 363])
    """
    spatial_dims = len(target_shape)
    pads = []
    for dim in reversed(range(spatial_dims)):
        in_size = input_tensor.shape[-spatial_dims + dim]
        out_size = target_shape[dim]
        total_pad = out_size - in_size
        pad_before = total_pad // 2
        pad_after = total_pad - pad_before
        pads.extend([pad_before, pad_after])
    # F.pad expects pads for last dim first
    padded_tensor = F.pad(input_tensor, pads, mode="constant", value=value)
    return padded_tensor


def center_unpad_nd(input_tensor, original_shape):
    """
    Center crops a tensor extracted from `input_tensor` to its original shape.

    This function works similarly to `center_pad_nd` and supports arbitrary spatial dimensions (e.g., 2D, 3D, etc.).

    Args:
        input_tensor: The padded tensor with shape `(N, C, *spatial_out)`.
        original_shape: The target spatial shape to which the tensor should be restored, specified as `(H, W)` or `(D, H, W)`, etc.

    Returns:
        cropped_tensor: The tensor cropped from the center region, with shape `(N, C, *original_shape)`.
    """
    spatial_dims = len(original_shape)
    slices = [slice(None), slice(None)]
    for dim in range(spatial_dims):
        out_size = input_tensor.shape[-spatial_dims + dim]
        crop_size = original_shape[dim]
        start = (out_size - crop_size) // 2
        end = start + crop_size
        slices.append(slice(start, end))
    cropped_tensor = input_tensor[tuple(slices)]
    return cropped_tensor


def add_sino_noise_guassian(measurement, level, noise_type="gaussian"):
    sino = measurement.squeeze().cpu().numpy()
    if noise_type == "gaussian":
        out = sino + np.random.normal(0, level, sino.shape)
    elif noise_type == "poisson":
        sino = level * np.exp(-sino * 0.01) + 10
        sino_noise = np.random.poisson(sino)
        sino_noise[sino_noise == 0] = 1
        out = -np.log(sino_noise / level) / 0.01

    SNR = 10 * np.log10(np.sum(sino**2) / np.sum((sino - out) ** 2))
    out = torch.tensor(out).unsqueeze(0).unsqueeze(0).float().to(measurement.device)

    return out, SNR


def fdk_reconstruct(gt_np, proj_geom, vol_geom):
    proj_id, sino = astra.create_sino3d_gpu(gt_np, proj_geom, vol_geom)
    rec_id = astra.data3d.create("-vol", vol_geom)

    cfg = astra.astra_dict("FDK_CUDA")
    cfg["ReconstructionDataId"] = rec_id
    cfg["ProjectionDataId"] = proj_id
    alg_id = astra.algorithm.create(cfg)

    astra.algorithm.run(alg_id, 1)

    recon = astra.data3d.get(rec_id)

    astra.algorithm.delete(alg_id)
    astra.data3d.delete(proj_id)
    astra.data3d.delete(rec_id)

    return recon


def astra_IR(gt_np, proj_geom, vol_geom, recon_algo="CGLS3D_CUDA", iter=10):
    proj_id, sino = astra.create_sino3d_gpu(gt_np, proj_geom, vol_geom)
    rec_id = astra.data3d.create("-vol", vol_geom)

    cfg = astra.astra_dict(recon_algo)
    cfg["ReconstructionDataId"] = rec_id
    cfg["ProjectionDataId"] = proj_id
    if recon_algo == "SIRT3D_CUDA":
        cfg["option"] = {"MinConstraint": 0.0}

    alg_id = astra.algorithm.create(cfg)

    astra.algorithm.run(alg_id, iter)

    recon = astra.data3d.get(rec_id)

    astra.algorithm.delete(alg_id)
    astra.data3d.delete(proj_id)
    astra.data3d.delete(rec_id)

    return recon
