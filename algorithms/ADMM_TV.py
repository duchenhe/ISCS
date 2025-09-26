"""ADMM-TV reconstructor class.

This module provides an ADMMTVReconstructor class equivalent to the
`get_ADMM_TV` function previously defined in `recon_solver.py`.

The class stores internal state (dual variables) and exposes a
callable interface via `__call__` so it can be used as a drop-in
replacement for the old factory-returned function.
"""

from typing import Any, Callable, Optional

import torch
from tqdm import tqdm


class ADMM_TV:
    """Class-based ADMM-TV reconstructor.

    Accepts a measurement model object that implements `A` and `A_T`.
    """

    def __init__(
        self,
        measure_model: Any,
        img_shape: Optional[torch.Size],
        lamb: float = 5.0,
        rho: float = 10.0,
        outer_iter: int = 30,
        inner_iter: int = 20,
        eps: float = 1e-10,
    ):
        self.measure_model = measure_model
        if img_shape is None:
            raise ValueError("img_shape must be provided to ADMM-TV")
        self.img_shape = img_shape
        self.lamb = lamb
        self.rho = rho
        self.outer_iter = outer_iter
        self.inner_iter = inner_iter
        self.eps = eps

        # dual/auxiliary variables (initialized on CPU, will be moved to device when used)
        self.del_x = torch.zeros(img_shape)
        self.del_y = torch.zeros(img_shape)
        self.udel_x = torch.zeros(img_shape)
        self.udel_y = torch.zeros(img_shape)

    def shrink(self, weight_src: torch.Tensor, lamb: float) -> torch.Tensor:
        return torch.sign(weight_src) * torch.max(torch.abs(weight_src) - lamb, torch.zeros_like(weight_src))

    def _Dx(self, x: torch.Tensor) -> torch.Tensor:
        y = torch.zeros_like(x)
        y[:, :, :-1, :] = x[:, :, 1:, :]
        y[:, :, -1, :] = x[:, :, 0, :]
        return y - x

    def _DxT(self, x: torch.Tensor) -> torch.Tensor:
        y = torch.zeros_like(x)
        y[:, :, :-1, :] = x[:, :, 1:, :]
        y[:, :, -1, :] = x[:, :, 0, :]
        tempt = -(y - x)
        difft = tempt[:, :, :-1, :]
        y[:, :, 1:, :] = difft
        y[:, :, 0, :] = x[:, :, -1, :] - x[:, :, 0, :]
        return y

    def _Dy(self, x: torch.Tensor) -> torch.Tensor:
        y = torch.zeros_like(x)
        y[:, :, :, :-1] = x[:, :, :, 1:]
        y[:, :, :, -1] = x[:, :, :, 0]
        return y - x

    def _DyT(self, x: torch.Tensor) -> torch.Tensor:
        y = torch.zeros_like(x)
        y[:, :, :, :-1] = x[:, :, :, 1:]
        y[:, :, :, -1] = x[:, :, :, 0]
        tempt = -(y - x)
        difft = tempt[:, :, :, :-1]
        y[:, :, :, 1:] = difft
        y[:, :, :, 0] = x[:, :, :, -1] - x[:, :, :, 0]
        return y

    def _A(self, x: torch.Tensor) -> torch.Tensor:
        return self.measure_model.A(x)

    def _AT(self, sinogram: torch.Tensor) -> torch.Tensor:
        return self.measure_model.A_T(sinogram)

    def A_cg(self, x: torch.Tensor) -> torch.Tensor:
        return self._AT(self._A(x)) + self.rho * (self._DxT(self._Dx(x)) + self._DyT(self._Dy(x)))

    def CG(
        self, A_fn: Callable[[torch.Tensor], torch.Tensor], b_cg: torch.Tensor, x: torch.Tensor, n_inner: int = 20
    ) -> torch.Tensor:
        r = b_cg - A_fn(x)
        p = r
        rs_old = torch.matmul(r.view(1, -1), r.view(1, -1).T)

        for i in range(n_inner):
            Ap = A_fn(p)
            a = rs_old / torch.matmul(p.view(1, -1), Ap.view(1, -1).T)

            x = x + a * p
            r = r - a * Ap

            rs_new = torch.matmul(r.view(1, -1), r.view(1, -1).T)
            if torch.sqrt(rs_new) < self.eps:
                break
            p = r + (rs_new / rs_old) * p
            rs_old = rs_new
        return x

    def CS_routine(self, x: torch.Tensor, ATy: torch.Tensor, niter: int = 30) -> torch.Tensor:
        # ensure state tensors are on same device as x
        if self.del_x.device != x.device:
            self.del_x = self.del_x.to(x.device)
            self.del_y = self.del_y.to(x.device)
            self.udel_x = self.udel_x.to(x.device)
            self.udel_y = self.udel_y.to(x.device)

        for _ in tqdm(range(niter)):
            b_cg = ATy + self.rho * (
                (self._DxT(self.del_x) - self._DxT(self.udel_x)) + (self._DyT(self.del_y) - self._DyT(self.udel_y))
            )
            x = self.CG(self.A_cg, b_cg, x, n_inner=self.inner_iter)

            self.del_x = self.shrink(self._Dx(x) + self.udel_x, self.lamb / self.rho)
            self.del_y = self.shrink(self._Dy(x) + self.udel_y, self.lamb / self.rho)
            self.udel_x = self._Dx(x) - self.del_x + self.udel_x
            self.udel_y = self._Dy(x) - self.del_y + self.udel_y

        return x

    def reconstruct(self, x_init: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Perform ADMM-TV update starting from x using measurement y.

        This mirrors the `ADMM_TV_fn` from the original function-based
        implementation.
        """
        ATy = self._AT(y)
        x = self.CS_routine(x_init, ATy, niter=self.outer_iter)
        return x
