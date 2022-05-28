import enum
import math

import numpy as np
import torch as th
from torch.cuda.amp import autocast
from torch.cuda.amp import GradScaler

from . import dist_util
from .nn import mean_flat
from .losses import normal_kl, discretized_gaussian_log_likelihood

class TimeDiffusion:
    def __init__(self):
        pass

    def training_losses(self, model, x_start, t, model_kwargs=None, noise=None):
        y = model_kwargs["y"]

        model_output = model(x_start)
        mse = (y - model_output) **2
        loss = (mse * ((y.detach()+2) **2)).mean(dim=list(range(1, len(y.shape))))
        mse = mse.mean(dim=list(range(1, len(y.shape))))

        terms = {}
        with th.no_grad():
            last_img_mse = ((y[:, 0] - x_start[:, -1]) **2).mean(dim=list(range(1, len(y.shape)-1)))
            #print(f"{mse[:,-1:]}, {last_img_mse}")
            #print(f"{mse[:,-1:]}, {last_img_mse}")
            terms["mse_base"] = last_img_mse
            terms["mse_ref"] = (mse.mean()/last_img_mse.mean())

        terms["mse_model"] = mse
        terms["loss"] = loss

        return terms
