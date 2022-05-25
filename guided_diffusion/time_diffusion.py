import enum
import math

import numpy as np
import torch as th
from torch.cuda.amp import autocast
from torch.cuda.amp import GradScaler

from .nn import mean_flat
from .losses import normal_kl, discretized_gaussian_log_likelihood

class TimeDiffusion:

    def training_losses(self, model, x_start, t, model_kwargs=None, noise=None):
        t = model_kwargs["t"]
        y = model_kwargs["y"]

        #with autocast():
        model_output = model(x_start, t)
        model_output = model_output.view((y.size()[0], -1)+y.size()[2:])
        mse = ((y - model_output) **2).mean(dim=list(range(2, len(y.shape))))
        #print(f"y {y.dtype}, model_output {model_output.dtype}, mse {mse.dtype}")

        with th.no_grad():
            ref_mse = ((y - x_start[:, -1:]) **2).mean(dim=list(range(2, len(y.shape))))

        terms = {}
        terms["mse"] = mse
        terms["ref_mse"] = ref_mse
        terms["loss"] = terms["mse"]
        #print(f"y.size()={y.size()}, model_output.size()={model_output.size()}, mse.size()={terms['mse'].size()}")

        return terms

