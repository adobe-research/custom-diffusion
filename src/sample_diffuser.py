# Copyright 2022 Adobe Research. All rights reserved.
# To view a copy of the license, visit LICENSE.md.

import argparse
import sys
import os
import numpy as np
import matplotlib.pyplot as plt
sys.path.append('./')
import torch
from diffusers import StableDiffusionPipeline
from src import diffuser_training 


def sample(ckpt, delta_ckpt, from_file, prompt, compress, freeze_model):
    model_id = ckpt
    pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float16).to("cuda")

    outdir = 'outputs/txt2img-samples'
    os.makedirs(outdir, exist_ok=True)
    if delta_ckpt is not None:
        diffuser_training.load_model(pipe.text_encoder, pipe.tokenizer, pipe.unet, delta_ckpt, compress, freeze_model)
        outdir = os.path.dirname(delta_ckpt)

    all_images = []
    if prompt is not None:
        images = pipe([prompt]*5, num_inference_steps=200, guidance_scale=6., eta=1.).images
        all_images += images
        images = np.hstack([np.array(x) for x in images])
        plt.imshow(images)
        plt.axis("off")
        plt.savefig(f'{outdir}/{prompt}.png', bbox_inches='tight')
    else:
        print(f"reading prompts from {from_file}")
        with open(from_file, "r") as f:
            data = f.read().splitlines()
            data = [5 * [prompt] for prompt in data]

        for prompt in data:
            images = pipe(prompt, num_inference_steps=200, guidance_scale=6., eta=1.).images
            all_images += images
            images = np.hstack([np.array(x) for x in images], 0)
            plt.imshow(images)
            plt.axis("off")
            plt.savefig(f'{outdir}/{prompt[0]}.png', bbox_inches='tight')

    os.makedirs(f'{outdir}/samples', exist_ok=True)
    for i, im in enumerate(all_images):
        im.save(f'{outdir}/samples/{i}.jpg')


def parse_args():
    parser = argparse.ArgumentParser('', add_help=False)
    parser.add_argument('--ckpt', help='target string for query',
                        type=str)
    parser.add_argument('--delta_ckpt', help='target string for query', default=None,
                        type=str)
    parser.add_argument('--from-file', help='path to prompt file', default='./',
                        type=str)
    parser.add_argument('--prompt', help='prompt to generate', default=None,
                        type=str)
    parser.add_argument("--compress", action='store_true')
    parser.add_argument('--freeze_model', help='crossattn or crossattn_kv', default='crossattn_kv',
                        type=str)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    sample(args.ckpt, args.delta_ckpt, args.from_file, args.prompt, args.compress, args.freeze_model)
