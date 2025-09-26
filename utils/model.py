import logging
import os
from pathlib import Path

import torch

from models import utils as mutils
from models.ema import ExponentialMovingAverage


def restore_checkpoint(ckpt_dir, state, device, skip_sigma=False, skip_optimizer=False):
    ckpt_dir = Path(ckpt_dir)
    if not ckpt_dir.exists():
        logging.warning(f"No checkpoint found at {ckpt_dir}. Returned the same state as input")
        return state
    else:
        loaded_state = torch.load(ckpt_dir, map_location=device, weights_only=False)
        if not skip_optimizer:
            state["optimizer"].load_state_dict(loaded_state["optimizer"])
        loaded_model_state = loaded_state["model"]
        if skip_sigma:
            loaded_model_state.pop("module.sigmas")

        state["model"].load_state_dict(loaded_model_state, strict=False)
        state["ema"].load_state_dict(loaded_state["ema"])
        state["step"] = loaded_state["step"]
        print(f"loaded checkpoint dir from {ckpt_dir}")
        return state


def get_score_models(config, ckpt_path):
    score_model = mutils.create_model(config)

    ema = ExponentialMovingAverage(score_model.parameters(), decay=config.model.ema_rate)
    state = dict(step=0, model=score_model, ema=ema)

    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"Checkpoint file {ckpt_path} is not found. Please check the path.")

    state = restore_checkpoint(ckpt_path, state, config.device, skip_optimizer=True)
    ema.copy_to(score_model.parameters())

    return score_model
