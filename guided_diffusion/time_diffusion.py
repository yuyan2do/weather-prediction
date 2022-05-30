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

    def training_losses(self, model, x, t, model_kwargs=None, noise=None):
        y = model_kwargs["y"]

        model_output = model(x)
        mse = (y - model_output) **2
        #weight = th.maximum(model_output.detach(), y.detach())+2
        weight = y.detach()+2
        # scale down loss for radar by 0.1
        weight[:,:2] = 0
        loss = mse * weight

        #loss = loss[:, -1:]
        #mse = mse

        terms = {}
        with th.no_grad():
            for i, category in enumerate(["precip", "radar", "wind"]):
                #print(f"y.size()={y.size()}")
                #print(f"x.size()={x.size()}")
                #print(f"{category} y={y[0, i-3]}")
                #print(f"{category} x={x[0, i-3]}")
                #print(f"{category} model_output={model_output[0, i-3]}")
                #print(f"{category} {(model_output[0, i-3]-x[0, i-3]).abs().sum()}")

                last_img_mse = ((y[:, i-3] - x[:, i-3]) **2).mean()
                #print(f"{mse[:,-1:]}, {last_img_mse}")
                #print(f"{mse[:,-1:]}, {last_img_mse}")
                terms["mse_" + category + "_base"] = last_img_mse
                mse_model = mse[:, i-3].mean()
                terms["mse_" + category + "_model"] = mse_model
                terms["mse_" + category + "_ref"] = (mse_model/last_img_mse)
                #print(f"{category}= {mse_model/last_img_mse}")

        terms["mse_model"] = mse.mean(dim=list(range(1, len(y.shape))))
        terms["loss"] = loss.mean(dim=list(range(1, len(y.shape))))

        return terms
