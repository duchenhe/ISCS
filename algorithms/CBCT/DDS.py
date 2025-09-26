from typing import Any, Dict, Optional

import torch
from tqdm import tqdm

import algorithms.utils as autils
from utils.result import save_nii_image
from models import utils as mutils
from pathlib import Path


class DDS:
    def __init__(self, sde: Any, model: Any, config: Any, measure_model: Any, eps: float = 1e-10):
        self.sde = sde
        self.model = model
        self.config = config
        self.measure_model = measure_model
        self.eps = eps

        # Precompute score function
        self.score_fn = mutils.get_score_fn(
            self.sde, self.model, train=False, continuous=self.config.training.continuous
        )

    def _A(self, x: torch.Tensor) -> torch.Tensor:
        return self.measure_model.A(x)

    def _AT(self, y: torch.Tensor) -> torch.Tensor:
        return self.measure_model.A_T(y)

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

    def denoise_update_fn(self, x):
        import sampling

        # Reverse diffusion predictor for denoising
        predictor_obj = sampling.ReverseDiffusionPredictor(self.sde, self.score_fn, probability_flow=False)
        vec_eps = torch.ones(x.shape[0], device=x.device) * self.eps
        _, x, _ = predictor_obj.update_fn(x, vec_eps)
        return x

    def reconstruct(
        self,
        x_init: torch.Tensor,
        y: torch.Tensor,
        cg_iter: int,
        w_dz: float,
        w_tik: float,
        renoise_method: str,
        save_path: Path,
        noise_control: Optional[Dict] = None,
        use_init: bool = False,
        save_intermediates: bool = False,
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

                # * ----------------------------------------------
                # * Batch processing for memory efficiency
                x_t_batch = autils.batchfy(x_t, 15)
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
                # * ----------------------------------------------

                pbar.set_postfix({"t": i, "sigma_t": float(sigma_t.item())})
                pbar.update()

                if save_intermediates:
                    save_nii_image(x_0_t, f"{save_path}/X_0/x_0_{i:04d}.nii.gz")

                # * ----------------------------------------------
                # * Data consistency
                x_0_t = x_0_t.movedim(0, 1)  # (300,1,256,256) -> (1,300,256,256)
                ATy = self._AT(y)
                rho = w_tik * (1 / sigma_t.item() ** 2)

                b_cg = ATy + rho * x_0_t

                x_0_t_hat = autils.cg_uni(self.A_cg_tik, b_cg, x_0_t, rho=rho, maxiter=cg_iter)

                x_0_t_hat = x_0_t_hat.movedim(0, 1)  # (1,300,256,256) -> (300,1,256,256)
                # * ----------------------------------------------

                if save_intermediates:
                    save_nii_image(x_0_t_hat, f"{save_path}/X_hat/DC/x_DC_{i:04d}.nii.gz")

                if i == sde.N - 1:
                    x_t = x_0_t_hat

                    x_t_batch = autils.batchfy(x_t, 15)
                    x_t_batches = []
                    for x_batch in x_t_batch:
                        x_batch = self.denoise_update_fn(x_batch)
                        x_t_batches.append(x_batch)
                        pbar_batch.update()

                    x_t = torch.cat(x_t_batches, dim=0)

                    break

                # * ----------------------------------------------
                # * Re-noising with noise control
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
                # * ----------------------------------------------

                if save_intermediates:
                    save_nii_image(x_t, f"{save_path}/X_hat/Re-noising/x_{i:04d}_addNoise.nii.gz")

        return x_t
