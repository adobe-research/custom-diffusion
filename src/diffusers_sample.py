# ==========================================================================================
#
# MIT License. To view a copy of the license, visit MIT_LICENSE.md.
#
# ==========================================================================================

import argparse
import sys
import os
import numpy as np
import torch
from PIL import Image

sys.path.append('./')
from src.diffusers_model_pipeline import CustomDiffusionPipeline


def sample(ckpt, delta_ckpt, from_file, prompt, compress, batch_size, freeze_model):
    model_id = ckpt
    pipe = CustomDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float16).to("cuda")
    pipe.load_model(delta_ckpt, compress)

    outdir = os.path.dirname(delta_ckpt)
    generator = torch.Generator(device='cuda').manual_seed(42)

    all_images = []
    if prompt is not None:
        images = pipe([prompt]*batch_size, num_inference_steps=200, guidance_scale=6., eta=1., generator=generator).images
        all_images += images
        images = np.hstack([np.array(x) for x in images])
        images = Image.fromarray(images)
        # takes only first 50 characters of prompt to name the image file
        name = '-'.join(prompt[:50].split())
        images.save(f'{outdir}/{name}.png')
    else:
        print(f"reading prompts from {from_file}")
        with open(from_file, "r") as f:
            data = f.read().splitlines()
            data = [[prompt]*batch_size for prompt in data]

        for prompt in data:
            images = pipe(prompt, num_inference_steps=200, guidance_scale=6., eta=1., generator=generator).images
            all_images += images
            images = np.hstack([np.array(x) for x in images], 0)
            images = Image.fromarray(images)
            # takes only first 50 characters of prompt to name the image file
            name = '-'.join(prompt[0][:50].split())
            images.save(f'{outdir}/{name}.png')

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
    parser.add_argument("--batch_size", default=5, type=int)
    parser.add_argument('--freeze_model', help='crossattn or crossattn_kv', default='crossattn_kv',
                        type=str)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    sample(args.ckpt, args.delta_ckpt, args.from_file, args.prompt, args.compress, args.batch_size, args.freeze_model)
