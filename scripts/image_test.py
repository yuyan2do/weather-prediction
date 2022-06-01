"""
Generate large batch of image samples from a model and save them as a large
numpy array. This can be used to produce samples for FID evaluation.
"""

import argparse
import os

from imageio.v2 import imread
from imageio.v2 import imwrite

import numpy as np
import torch as th
import torch.distributed as dist

from pathlib import Path

from guided_diffusion import dist_util, logger
from guided_diffusion.script_util import (
    NUM_CLASSES,
    model_and_diffusion_defaults,
    create_model_and_diffusion,
    add_dict_to_argparser,
    args_to_dict,
)


def main():
    args = create_argparser().parse_args()

    dist_util.setup_dist()
    logger.configure()

    logger.log("creating model and diffusion...")
    model, diffusion = create_model_and_diffusion(
        **args_to_dict(args, model_and_diffusion_defaults().keys())
    )
    model.load_state_dict(
        dist_util.load_state_dict(args.model_path, map_location="cpu")
    )
    model.to(dist_util.dev())
    if args.use_fp16:
        model.convert_to_fp16()
    model.eval()

    def get_item(f, i):
        if True:
            arr = []
            for img_idx in range(i+1, i+5*4, 5):
                if img_idx <= 20:
                    for category in ["precip", "radar", "wind"]:
                        fname = os.path.join(args.data_dir, category.capitalize(), f, f"{category}_{img_idx:03d}.png")
                        img = np.asarray(imread(fname)).astype(np.float32) / 127.5 - 1
                        arr.append(img)
                else:
                    for category in ["precip", "radar", "wind"]:
                        img_idx2 = img_idx - 20
                        fname = os.path.join(args.output_dir, category.capitalize(), f, f"{category}_{img_idx2:03d}.png")
                        # print(f"read fname={fname}")
                        img = np.asarray(imread(fname)).astype(np.float32) / 127.5 - 1
                        arr.append(img)

            arr = np.stack(arr, axis=0)
            return arr

    def save_img(f, img_idx, y):
        #print(f"args.output_dir={args.output_dir}")
        #path = os.path.join(args.output_dir, f)
        #Path(path).mkdir(parents=True, exist_ok=True)
        #print(f"path={path}")
        #fname = os.path.join(path, f"wind_{i:03d}.png")
        for i, category in enumerate(["precip", "radar", "wind"]):
            path = os.path.join(args.output_dir, category.capitalize(), f)
            Path(path).mkdir(parents=True, exist_ok=True)
            fname = os.path.join(path, f"{category}_{img_idx:03d}.png")
            #print(f"fname={fname}")
            imwrite(fname, y[i])

    for f in sorted(os.listdir(os.path.join(args.data_dir, "Wind"))):
        d = os.path.join(args.data_dir, "Wind", f)
        if os.path.isdir(d):
            #print(sample.shape)
            print(f"start eval {d}")

            for i in range(20):
                sample = np.expand_dims(get_item(f, i), axis=0)
                x = th.from_numpy(sample).to(dist_util.dev())

                with th.no_grad():
                    model_output = model(x)
                    # last img
                    # print(f"model_output {model_output.size()}")
                    model_output = model_output[0]

                y = model_output.cpu().detach().numpy()
                y = (y + 1) * 127.5
                y = np.clip(y, 0, 255).astype(np.uint8)
                save_img(f, i+1, y)


def create_argparser():
    defaults = dict(
        data_dir="",
        output_dir="",
        clip_denoised=True,
        num_samples=10000,
        batch_size=16,
        use_ddim=False,
        model_path="",
    )
    defaults.update(model_and_diffusion_defaults())
    parser = argparse.ArgumentParser()
    add_dict_to_argparser(parser, defaults)
    return parser


if __name__ == "__main__":
    main()
