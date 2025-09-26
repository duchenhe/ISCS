from pathlib import Path
from typing import Any, Dict, Optional

import torch
from tqdm import tqdm

import algorithms.utils as autils
from models import utils as mutils
from utils.result import save_nii_image


class DDS:
    def __init__(self, sde: Any, model: Any, config: Any, measure_model: Any, eps: float = 1e-10, factor: int = 4):
        self.sde = sde
        self.model = model
        self.config = config
        self.measure_model = measure_model
        self.eps = eps

        # Precompute score function
        self.score_fn = mutils.get_score_fn(
            self.sde, self.model, train=False, continuous=self.config.training.continuous
        )
        self.factor = factor
        self.M = autils.get_M(factor)
        self.invM = torch.inverse(self.M)

    def _A(self, x: torch.Tensor) -> torch.Tensor:
        return self.measure_model.A(x)

    def _AT(self, y: torch.Tensor) -> torch.Tensor:
        ATy = self.measure_model.A_dagger(y)
        ATy = self.measure_model.couple(self.measure_model.decouple(ATy))
        return ATy

    def A_cg(self, x: torch.Tensor) -> torch.Tensor:
        return self._AT(self._A(x))

    def A_cg_tik(self, x: torch.Tensor, rho: float) -> torch.Tensor:
        return self._AT(self._A(x)) + rho * x

    def tweedie_denoising_fn(self, x: torch.Tensor, t: torch.Tensor) -> tuple:
        vec_t = torch.ones(x.shape[0], device=x.device) * t
        with torch.no_grad():
            score_t = self.score_fn(x, vec_t)
        sigma_t = self.sde.sigma_min * (self.sde.sigma_max / self.sde.sigma_min) ** t
        x_0_hat = x + (sigma_t**2) * score_t
        return x_0_hat, score_t, sigma_t

    def reconstruct(
        self,
        x_init: torch.Tensor,
        y: torch.Tensor,
        save_path: Path,
        cg_iter: int,
        w_dz: float = 0.5,
        noise_control: Optional[Dict] = None,
        use_init: bool = False,
        save_intermediates: bool = False,
        renoise_method: str = "DDPM",
    ) -> torch.Tensor:
        sde = self.sde
        eps = self.eps

        with torch.no_grad():
            timesteps = torch.linspace(sde.T, eps, sde.N)

            if use_init:
                noises = torch.randn_like(x_init).to(y.device)

                t = timesteps[0]
                x_t = autils.re_noising(
                    renoise_method="DDPM",
                    x_0_t=x_init,
                    score_t=None,
                    noises=noises,
                    sde=sde,
                    t_curr=t,
                    t_next=t,
                    noise_control=noise_control,
                )
            else:
                x_t = sde.prior_sampling(x_init.shape).to(y.device)

            pbar = tqdm(total=sde.N, desc="(0. DM)", colour="red", leave=True)

            for i in range(sde.N):
                t = timesteps[i]

                x_t_batch = autils.batchfy(x_t, 20)
                x_t_batches = []
                score_t_batches = []
                sigma_t = torch.tensor(0.0, device=y.device)

                pbar_batch = tqdm(total=len(x_t_batch), desc="(1. Batch)", colour="green", position=1, leave=False)

                for x_batch in x_t_batch:
                    x_batch, score_t_batch, sigma_t = self.tweedie_denoising_fn(x_batch, t)
                    x_t_batches.append(x_batch)
                    score_t_batches.append(score_t_batch)
                    pbar_batch.update()

                x_0_t = torch.cat(x_t_batches, dim=0)
                score_t = torch.cat(score_t_batches, dim=0)

                pbar.set_postfix({"t": i, "sigma_t": float(sigma_t.item())})
                pbar.update()

                if save_intermediates:
                    save_nii_image(x_0_t, f"{save_path}/X_0/x_0_{i:04d}.nii.gz")

                # * ---------------------------------------------
                # * Data consistency
                ATy = self._AT(y)

                x_0_t_hat = autils.cg_uni(self.A_cg_tik, ATy, x_0_t, rho=0, maxiter=cg_iter, tol=1e-8)

                # * ---------------------------------------------

                if i == sde.N - 1:
                    x_t = x_0_t_hat
                    break

                noises = torch.randn_like(x_0_t_hat).to(y.device)

                x_t = autils.re_noising(
                    renoise_method=renoise_method,
                    x_0_t=x_0_t_hat,
                    score_t=score_t,
                    noises=noises,
                    sde=sde,
                    t_curr=timesteps[i],
                    t_next=timesteps[i + 1],
                    noise_control=noise_control,
                )

                if save_intermediates:
                    save_nii_image(x_t, f"{save_path}/X_hat/Re-noising/x_{i:04d}_addNoise.nii.gz")

        return x_t
