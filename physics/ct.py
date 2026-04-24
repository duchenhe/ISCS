import numpy as np
import torch
import torch_radon

from utils.data import center_pad_nd, center_unpad_nd

PI = 4 * torch.ones(1).atan()


class PBCT_carterbox:
    def __init__(self, det_count, view_available, view_full_num, recon_size):
        """

        Args:
            det_count (_type_): Number of rays to be projected. The number of rays projected per view.
            view_available (_type_): Number of visible views (if limited-view CT is involved)
            view_full_num (_type_): Total number of views
        """
        self.det_count = det_count
        self.angles_FV = np.linspace(0, np.pi, view_full_num, endpoint=False)
        self.angles_LV = self.angles_FV[view_available]
        self.recon_size = recon_size
        self.view_full_num = view_full_num
        self.LV_list = view_available

        self.radon_FV = torch_radon.ParallelBeam(det_count=det_count, angles=self.angles_FV)
        self.radon_LV = torch_radon.ParallelBeam(det_count=det_count, angles=self.angles_LV)

    # * Main
    # A: Radon Transform
    def A(self, x):
        if x.shape[2] != self.det_count:
            x = center_pad_nd(x, (self.det_count, self.det_count), value=0.0).float()

        sino = self.radon_LV.forward(x)
        return sino

    # A_T
    def A_T(self, y):
        recon = self.radon_LV.backward(y)
        recon = center_unpad_nd(recon, (self.recon_size, self.recon_size))

        norm_factor = PI.item() / (2 * len(self.angles_LV))
        recon = recon * norm_factor

        return recon

    # Moore–Penrose （pseudo-inverse）: FBP
    def A_dagger(self, y, filter_name="ramp"):
        sino_filtered = self.radon_LV.filter_sinogram(y, filter_name=filter_name)
        recon = self.radon_LV.backward(sino_filtered)
        recon = center_unpad_nd(recon, (self.recon_size, self.recon_size))

        return recon

    # * ------------------------------------------

    def A_FV(self, x):
        if x.shape[2] != self.det_count:
            x = center_pad_nd(x, (self.det_count, self.det_count), value=0.0)

        sino = self.radon_FV.forward(x)
        return sino

    def FBP_FV(self, y, filter_name="ramp"):
        sino_filtered = self.radon_FV.filter_sinogram(y, filter_name=filter_name)
        recon = self.radon_FV.backward(sino_filtered)
        recon = center_unpad_nd(recon, (self.recon_size, self.recon_size))

        return recon

    def BP_FV(self, y):
        recon = self.radon_FV.backward(y)
        recon = center_unpad_nd(recon, (self.recon_size, self.recon_size))

        norm_factor = PI.item() / (2 * len(self.angles_FV))
        recon = recon * norm_factor

        return recon

    def get_angles(self):
        return self.angles_FV, self.angles_LV


class CBCT_carterbox:
    def __init__(
        self,
        angles_FV,
        angle_LV,
        det_count_u,
        det_count_v,
        det_spacing_u,
        det_spacing_v,
        src_dist=300,
        det_dist=300,
        pitch=0,
        base_z=0,
        volume=None,
    ):
        """

        Args:
            det_count (_type_): Number of rays to be projected. The number of rays projected per view.
            view_available (_type_): Number of visible views (if limited-view CT is involved)
            view_full_num (_type_): Total number of views
        """
        self.angles_FV = angles_FV
        self.angles_LV = angle_LV

        self.det_count_u = det_count_u
        self.det_count_v = det_count_v
        self.det_spacing_u = det_spacing_u
        self.det_spacing_v = det_spacing_v
        self.src_dist = src_dist
        self.det_dist = det_dist
        self.volume = volume

        self.radon_FV = torch_radon.ConeBeam(
            det_count_u=self.det_count_u,
            angles=self.angles_FV,
            src_dist=self.src_dist,
            det_dist=self.det_dist,
            det_count_v=self.det_count_v,
            det_spacing_u=self.det_spacing_u,
            det_spacing_v=self.det_spacing_v,
            pitch=pitch,
            base_z=base_z,
            volume=self.volume,
        )

        self.radon_LV = torch_radon.ConeBeam(
            det_count_u=self.det_count_u,
            angles=self.angles_LV,
            src_dist=self.src_dist,
            det_dist=self.det_dist,
            det_count_v=self.det_count_v,
            det_spacing_u=self.det_spacing_u,
            det_spacing_v=self.det_spacing_v,
            pitch=pitch,
            base_z=base_z,
            volume=self.volume,
        )

    def A(self, x):
        sino = self.radon_LV.forward(x)
        return sino

    def A_T(self, y):
        recon = self.radon_LV.backward(y)
        return recon

    def A_FV(self, x: torch.Tensor):
        sino = self.radon_FV.forward(x)
        return sino

    def _fdk_preweight(self, y):
        if y.dim() != 4:
            raise ValueError(f"CBCT projection must have shape [B, angles, det_v, det_u], got {tuple(y.shape)}")

        det_v = y.shape[-2]
        det_u = y.shape[-1]
        device = y.device
        dtype = y.dtype

        u = (torch.arange(det_u, device=device, dtype=dtype) - (det_u - 1) / 2) * self.det_spacing_u
        v = (torch.arange(det_v, device=device, dtype=dtype) - (det_v - 1) / 2) * self.det_spacing_v
        vv, uu = torch.meshgrid(v, u, indexing="ij")

        source_to_detector = self.src_dist + self.det_dist
        weight = source_to_detector / torch.sqrt(source_to_detector**2 + uu**2 + vv**2)
        weight = weight * (source_to_detector / self.src_dist)
        return y * weight.view(1, 1, det_v, det_u)

    def _filter_cbct_sinogram(self, y, radon, filter_name):
        if y.dim() != 4:
            raise ValueError(f"CBCT projection must have shape [B, angles, det_v, det_u], got {tuple(y.shape)}")

        batch, n_angles, det_v, det_u = y.shape
        y = y.permute(0, 2, 1, 3).reshape(batch * det_v, n_angles, det_u)
        y = radon.filter_sinogram(y, filter_name=filter_name)
        y = y.reshape(batch, det_v, n_angles, det_u).permute(0, 2, 1, 3).contiguous()

        # The ramp filter is defined on physical detector coordinates.
        return y / self.det_spacing_u

    def _fdk(self, y, radon, filter_name):
        y = self._fdk_preweight(y)
        y = self._filter_cbct_sinogram(y, radon, filter_name)
        recon = radon.backward(y)

        source_to_detector = self.src_dist + self.det_dist
        bp_scale = self.det_spacing_u * self.det_spacing_v * (self.src_dist / source_to_detector) ** 2
        return recon * bp_scale

    def FDK_LV(self, y, filter_name="ramp"):
        return self._fdk(y, self.radon_LV, filter_name)

    def FDK_FV(self, y, filter_name="ramp"):
        return self._fdk(y, self.radon_FV, filter_name)

    def BP_FV(self, y):
        recon = self.radon_FV.backward(y)
        return recon

    def get_angles(self):
        return self.angles_FV, self.angles_LV
