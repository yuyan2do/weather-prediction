"""
Generate a large batch of image samples from a model and save them as a large
numpy array. This can be used to produce samples for FID evaluation.
"""

import argparse
import os

from imageio import imread
from imageio import imwrite

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
            for img_idx in range(i+1, 21, 1):
                path = os.path.join(args.data_dir, f)
                fname = os.path.join(path, f"wind_{img_idx:03d}.png")
                img = np.asarray(imread(fname)).astype(np.float32) / 127.5 - 1
                arr.append(np.expand_dims(img, axis=0))
            for img_idx in range(1, i+1, 1):
                path = os.path.join(args.output_dir, f)
                fname = os.path.join(path, f"wind_{i:03d}.png")
                #fname = os.path.join(folder, f"wind_{img_idx:03d}.png")
                img = np.asarray(imread(fname)).astype(np.float32) / 127.5 - 1
                arr.append(np.expand_dims(img, axis=0))

            arr = np.stack(arr, axis=-1)
            return np.transpose(arr, [3, 0, 1, 2])

    def save_img(f, i, y):
        #print(f"args.output_dir={args.output_dir}")
        path = os.path.join(args.output_dir, f)
        Path(path).mkdir(parents=True, exist_ok=True)
        #print(f"path={path}")
        fname = os.path.join(path, f"wind_{i:03d}.png")
        #print(f"fname={fname}")
        imwrite(fname, y)

    timestep = []
    for i in range(0, 40):
        timestep.append(th.arange(i, i+20, 1).to(dist_util.dev()))

    for f in sorted(os.listdir(args.data_dir)):
        d = os.path.join(args.data_dir, f)
        if os.path.isdir(d):
            #print(sample.shape)
            print(f"start eval {d}")

            for i in range(20):
                sample = np.expand_dims(get_item(f, i), axis=0)
                x = th.from_numpy(sample).to(dist_util.dev())

                t = timestep[i]

                with th.no_grad():
                    model_output = model(x, t)
                    # last img
                    # print(f"model_output {model_output.size()}")
                    model_output = model_output[-1,0]

                y = model_output.cpu().detach().numpy()
                y = (y + 1) * 127.5
                y = np.clip(y, 0, 255).astype(np.uint8)
                save_img(f, i+1, y)

    return

    logger.log("sampling...")
    all_images = []
    all_labels = []
    while len(all_images) * args.batch_size < args.num_samples:
        model_kwargs = {}
        if args.class_cond:
            classes = th.randint(
                low=0, high=NUM_CLASSES, size=(args.batch_size,), device=dist_util.dev()
            )
            model_kwargs["y"] = classes
        sample_fn = (
            diffusion.p_sample_loop if not args.use_ddim else diffusion.ddim_sample_loop
        )
        sample = sample_fn(
            model,
            (args.batch_size, 3, args.image_size, args.image_size),
            clip_denoised=args.clip_denoised,
            model_kwargs=model_kwargs,
        )
        sample = ((sample + 1) * 127.5).clamp(0, 255).to(th.uint8)
        sample = sample.permute(0, 2, 3, 1)
        sample = sample.contiguous()

        gathered_samples = [th.zeros_like(sample) for _ in range(dist.get_world_size())]
        dist.all_gather(gathered_samples, sample)  # gather not supported with NCCL
        all_images.extend([sample.cpu().numpy() for sample in gathered_samples])
        if args.class_cond:
            gathered_labels = [
                th.zeros_like(classes) for _ in range(dist.get_world_size())
            ]
            dist.all_gather(gathered_labels, classes)
            all_labels.extend([labels.cpu().numpy() for labels in gathered_labels])
        logger.log(f"created {len(all_images) * args.batch_size} samples")

    arr = np.concatenate(all_images, axis=0)
    arr = arr[: args.num_samples]
    if args.class_cond:
        label_arr = np.concatenate(all_labels, axis=0)
        label_arr = label_arr[: args.num_samples]
    if dist.get_rank() == 0:
        shape_str = "x".join([str(x) for x in arr.shape])
        out_path = os.path.join(logger.get_dir(), f"samples_{shape_str}.npz")
        logger.log(f"saving to {out_path}")
        if args.class_cond:
            np.savez(out_path, arr, label_arr)
        else:
            np.savez(out_path, arr)

    dist.barrier()
    logger.log("sampling complete")


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
