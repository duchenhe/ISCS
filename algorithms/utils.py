import math

import numpy as np
import torch


def batchfy(tensor, batch_size):
    n = len(tensor)
    num_batches = math.ceil(n / batch_size)
    return tensor.chunk(num_batches, dim=0)


def cg_uni(A_fn, b, x=None, rho=0.0, maxiter=50, tol=1e-5):
    if x is None:
        x = torch.zeros_like(b)

    # r = b - (A+rhoI)x
    r = b - A_fn(x, rho)
    p = r.clone()

    def dot(u, v):
        # return (u * v).sum()
        return torch.sum(u.conj() * v)

    rs_old = dot(r, r)

    for _ in range(maxiter):
        Ap = A_fn(p, rho)
        denom = dot(p, Ap)

        if torch.abs(denom) < 1e-30:
            break

        alpha = rs_old / denom
        x = x + alpha * p
        r = r - alpha * Ap

        rs_new = dot(r, r)

        if rs_new < tol**2:
            break

        beta = rs_new / rs_old
        p = r + beta * p
        rs_old = rs_new

    return x


def re_noising(
    x_0_t,
    score_t,
    noises,
    renoise_method="DDPM",
    sde=None,
    t_curr=None,
    t_next=None,
    noise_control=None,
):
    """
    The logic for computing x_t, which encapsulates different resampling methods.

    **Parameters:**
    - `resampling_method` (str): The sampling method, supporting "DDS", "DiffPIR", "DDIM", or "DDPM".
    - `x_0_t` (Tensor): The estimated value at the current time step.
    - `score_t` (Tensor): The score at the current time step.
    - `noise` (Tensor): Random noise.
    - `sde`: An SDE object.
    - `t_curr`: The current time step.
    - `t_next`: The next time step.

    **Returns:**
    - `x_t` (Tensor): The sampled x_t after applying the resampling method.
    """
    if noise_control == "SLERP":
        z_0 = torch.randn_like(x_0_t[0].unsqueeze(0))
        z_1 = torch.randn_like(x_0_t[0].unsqueeze(0))
        if torch.all(z_0 == z_1):
            print("z_0 and z_1 are the same, using only one noise sample.")

        noises = slerp_path(z_0, z_1, n_mid=x_0_t.shape[0] * 16, include_endpoints=False)
        noises, idx = take_from_center(noises, n=x_0_t.shape[0], step_left=16, step_right=16)
    elif noise_control == "None":
        pass
    else:
        raise ValueError(f"unknown noise_control: {noise_control}")

    sigma_t_curr = sde.sigma_min * (sde.sigma_max / sde.sigma_min) ** t_curr
    sigma_t_next = sde.sigma_min * (sde.sigma_max / sde.sigma_min) ** t_next

    if renoise_method == "DDS":
        eta = 0.85

        sigma_sto = eta * sigma_t_next
        sigma_det = math.sqrt(1 - eta**2) * sigma_t_next

        noise_sto = sigma_sto * noises
        noise_det = sigma_det * (-sigma_t_curr) * score_t

        x_t = x_0_t + noise_det + noise_sto

    elif renoise_method == "DiffPIR":
        eta = 0.5

        sigma_sto = eta * sigma_t_next
        sigma_det = (1 - eta) * sigma_t_next

        noise_sto = sigma_sto * noises
        noise_det = sigma_det * (-sigma_t_curr) * score_t

        x_t = x_0_t + noise_det + noise_sto

    elif renoise_method == "DDIM":
        eta = 1

        sigma_sto = eta * math.sqrt(sigma_t_curr**2 - sigma_t_next**2)
        sigma_det = math.sqrt(max(0, sigma_t_next**2 - sigma_sto**2))

        noise_sto = sigma_sto * noises
        noise_det = sigma_det * (-sigma_t_curr) * score_t

        x_t = x_0_t + noise_det + noise_sto

    elif renoise_method == "DDPM":
        x_t = x_0_t + sigma_t_next * noises

    else:
        raise ValueError(f"unknown resampling_method: {renoise_method}")

    x_t = x_t.detach()

    return x_t


def get_M(factor: int):
    if factor == 2:
        return torch.tensor(([[0.70710678118, 0.70710678118], [0.70710678118, -0.70710678118]]))
    elif factor == 3:
        return torch.tensor(
            [
                [5.7735014e-01, -8.1649649e-01, 4.7008697e-08],
                [5.7735026e-01, 4.0824834e-01, 7.0710671e-01],
                [5.7735026e-01, 4.0824822e-01, -7.0710683e-01],
            ]
        )
    elif factor == 4:
        return torch.tensor(
            [
                [0.5, 0.866025403784439, 0, 0],
                [0.5, -0.288675134594813, 0.816496580927726, 0],
                [0.5, -0.288675134594813, -0.408248290463863, 0.707106781186548],
                [0.5, -0.288675134594813, -0.408248290463863, -0.707106781186548],
            ]
        )
    elif factor == 5:
        return torch.tensor(
            [
                [0.447213595499958, 0.894427190999916, 0, 0, 0],
                [0.447213595499958, -0.223606797749979, 0.866025403784439, 0, 0],
                [0.447213595499958, -0.223606797749979, -0.288675134594813, 0.816496580927726, 0],
                [0.447213595499958, -0.223606797749979, -0.288675134594813, -0.408248290463863, 0.707106781186548],
                [0.447213595499958, -0.223606797749979, -0.288675134594813, -0.408248290463863, -0.707106781186548],
            ]
        )
    else:
        raise ValueError(f"unsupported zsr-factor ({factor}) for kernel")


def slerp_path(
    z0: torch.Tensor,
    z1: torch.Tensor,
    n_mid: int = 30,
    include_endpoints: bool = False,
    eps: float = 1e-8,
):
    """
    Use SLERP to generate a smooth path between z1 and z2.

    Parameters:
        z1, z2: Tensors with arbitrary dimensions (e.g., [1, 1, 256, 256]), as long as they have the same shape.
        n_mid: Number of intermediate interpolation points to generate.
        include_endpoints: Whether to include z1 and z2 themselves in the output.
        eps: A small value added to prevent numerical instability.

    Returns:
        Tensor with shape [N, *z_shape].
        If `include_endpoints=False`, then N = n_mid.
        If `include_endpoints=True`, then N = n_mid + 2, with the first and last elements being z1 and z2, respectively.
    """
    assert z0.shape == z1.shape, "z0 and z1 must have the same shape."
    # Calculate the angle after flattening
    z0_flat = z0.reshape(-1)
    z1_flat = z1.reshape(-1)
    # Compute the inner product and norm
    dot = torch.dot(z0_flat, z1_flat)
    norm_prod = z0_flat.norm() * z1_flat.norm() + eps
    cos_theta = torch.clamp(dot / norm_prod, -1.0 + 1e-7, 1.0 - 1e-7)
    theta = torch.acos(cos_theta)

    if theta < 1e-6 or torch.abs(theta - np.pi) < 1e-6:
        print("Warning: z0 and z1 are too close or opposite, using linear interpolation.")
        num = n_mid + 2
        alphas = torch.linspace(0, 1, num, device=z0.device, dtype=z0.dtype)
        if not include_endpoints:
            alphas = alphas[1:-1]
        return torch.stack([(1 - a) * z0 + a * z1 for a in alphas]).squeeze(1)

    num = n_mid + 2
    alphas = torch.linspace(0, 1, num, device=z0.device, dtype=z0.dtype)
    if not include_endpoints:
        alphas = alphas[1:-1]

    sin_theta = torch.sin(theta)
    outs = []
    for a in alphas:
        w1 = torch.sin((1 - a) * theta) / (sin_theta + eps)
        w2 = torch.sin(a * theta) / (sin_theta + eps)
        z = w1 * z0 + w2 * z1
        outs.append(z)

    return torch.stack(outs).squeeze(1)


def take_from_center(t: torch.Tensor, n: int, step_left: int = 1, step_right: int = 1, dim: int = 0):
    """
    Samples `n` slices from dimension `dim` of tensor `t`, following a "center-to-both-sides" sampling pattern.

    Parameters
    ----------
    t           : Input tensor, with shape [L, C, H, W] or other
    n           : Total number of slices to sample
    step_left   : Step size for sampling toward the left (negative direction)
    step_right  : Step size for sampling toward the right (positive direction)
    dim         : Dimension along which sampling is performed; default is 0

    Returns
    -------
    out         : Sampled tensor, with shape [n, C, H, W]
    idx_sorted  : Indices of the actual slices retrieved (sorted)
    """
    L = t.size(dim)
    if n > L:
        raise ValueError("n cannot exceed the length of that dimension.")

    center = L // 2
    indices = [center]

    k = 1
    while len(indices) < n and (center - k * step_left >= 0 or center + k * step_right < L):
        # left
        left = center - k * step_left
        if left >= 0:
            indices.append(left)
            if len(indices) == n:
                break
        # right
        right = center + k * step_right
        if right < L and len(indices) < n:
            indices.append(right)
        k += 1

    idx_sorted = sorted(indices)
    out = t.index_select(dim, torch.tensor(idx_sorted, device=t.device))
    return out, idx_sorted
