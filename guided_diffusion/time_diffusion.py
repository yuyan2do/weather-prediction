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
        # only compute loss for image from 11 to 20
        self.mse_mask = th.FloatTensor([0]*10+[1]*10)[None, :].to(dist_util.dev())
        #self.mse_mask.requires_grad_(False)


    def training_losses(self, model, x_start, t, model_kwargs=None, noise=None):
        t = model_kwargs["t"]
        y = model_kwargs["y"]

        #with autocast():
        model_output = model(x_start, t)
        model_output = model_output.view((y.size()[0], -1)+y.size()[2:])
        mse = ((y - model_output) **2).mean(dim=list(range(2, len(y.shape))))

        mse = mse * self.mse_mask
        #print(f"y {y.dtype}, model_output {model_output.dtype}, mse {mse.dtype} {mse.size()}")

        terms = {}
        with th.no_grad():
            # baseline is using 19th image
            last_img_mse = ((y - x_start[:, -2:-1]) **2).mean(dim=list(range(2, len(y.shape))))
            #print(f"{mse[:,-1:]}, {last_img_mse}")
            terms["ref_mse"] = mse[:, -1:]/last_img_mse

        terms["mse"] = mse
        terms["loss"] = terms["mse"]
        #print(f"y.size()={y.size()}, model_output.size()={model_output.size()}, mse.size()={terms['mse'].size()}")

        return terms

