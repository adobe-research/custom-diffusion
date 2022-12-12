# Copyright 2019 Adobe. All rights reserved.
# Creative Commons Attribution-NonCommercial-ShareAlike
# 4.0 International Public License (CC-NC-SA-4.0). To view a copy of the license, visit
# https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode.

import os
import argparse
import glob
import torch

layers = [
 'model.diffusion_model.input_blocks.1.1.transformer_blocks.0.attn2.to_k.weight',
 'model.diffusion_model.input_blocks.1.1.transformer_blocks.0.attn2.to_v.weight',
 'model.diffusion_model.input_blocks.2.1.transformer_blocks.0.attn2.to_k.weight',
 'model.diffusion_model.input_blocks.2.1.transformer_blocks.0.attn2.to_v.weight',
 'model.diffusion_model.input_blocks.4.1.transformer_blocks.0.attn2.to_k.weight',
 'model.diffusion_model.input_blocks.4.1.transformer_blocks.0.attn2.to_v.weight',
 'model.diffusion_model.input_blocks.5.1.transformer_blocks.0.attn2.to_k.weight',
 'model.diffusion_model.input_blocks.5.1.transformer_blocks.0.attn2.to_v.weight',
 'model.diffusion_model.input_blocks.7.1.transformer_blocks.0.attn2.to_k.weight',
 'model.diffusion_model.input_blocks.7.1.transformer_blocks.0.attn2.to_v.weight',
 'model.diffusion_model.input_blocks.8.1.transformer_blocks.0.attn2.to_k.weight',
 'model.diffusion_model.input_blocks.8.1.transformer_blocks.0.attn2.to_v.weight',
 'model.diffusion_model.middle_block.1.transformer_blocks.0.attn2.to_k.weight',
 'model.diffusion_model.middle_block.1.transformer_blocks.0.attn2.to_v.weight',
 'model.diffusion_model.output_blocks.3.1.transformer_blocks.0.attn2.to_k.weight',
 'model.diffusion_model.output_blocks.3.1.transformer_blocks.0.attn2.to_v.weight',
 'model.diffusion_model.output_blocks.4.1.transformer_blocks.0.attn2.to_k.weight',
 'model.diffusion_model.output_blocks.4.1.transformer_blocks.0.attn2.to_v.weight',
 'model.diffusion_model.output_blocks.5.1.transformer_blocks.0.attn2.to_k.weight',
 'model.diffusion_model.output_blocks.5.1.transformer_blocks.0.attn2.to_v.weight',
 'model.diffusion_model.output_blocks.6.1.transformer_blocks.0.attn2.to_k.weight',
 'model.diffusion_model.output_blocks.6.1.transformer_blocks.0.attn2.to_v.weight',
 'model.diffusion_model.output_blocks.7.1.transformer_blocks.0.attn2.to_k.weight',
 'model.diffusion_model.output_blocks.7.1.transformer_blocks.0.attn2.to_v.weight',
 'model.diffusion_model.output_blocks.8.1.transformer_blocks.0.attn2.to_k.weight',
 'model.diffusion_model.output_blocks.8.1.transformer_blocks.0.attn2.to_v.weight',
 'model.diffusion_model.output_blocks.9.1.transformer_blocks.0.attn2.to_k.weight',
 'model.diffusion_model.output_blocks.9.1.transformer_blocks.0.attn2.to_v.weight',
 'model.diffusion_model.output_blocks.10.1.transformer_blocks.0.attn2.to_k.weight',
 'model.diffusion_model.output_blocks.10.1.transformer_blocks.0.attn2.to_v.weight',
 'model.diffusion_model.output_blocks.11.1.transformer_blocks.0.attn2.to_k.weight',
 'model.diffusion_model.output_blocks.11.1.transformer_blocks.0.attn2.to_v.weight'
]


def main(path):
    for files in glob.glob(f'{path}/checkpoints/*'):
        if ('=' in files or '_' in files) and 'delta' not in files:
            print(files)
            if '=' in files:
                epoch_number = files.split('=')[1].split('.ckpt')[0]
            elif '_' in files:
                epoch_number = files.split('/')[-1].split('.ckpt')[0]

            st = torch.load(files)["state_dict"]
            st_delta = {'state_dict': {}}
            for each in layers:
                st_delta['state_dict'][each] = st[each].clone()
            print('/'.join(files.split('/')[:-1]) + f'/delta_epoch={epoch_number}.ckpt')

            num_tokens = st['cond_stage_model.transformer.text_model.embeddings.token_embedding.weight'].shape[0]

            if num_tokens > 49408:
                print("$$$$$$$$$ saving the optimized embedding")
                st_delta['state_dict']['embed'] = st['cond_stage_model.transformer.text_model.embeddings.token_embedding.weight'][-(num_tokens-49408):].clone()
                print(st_delta['state_dict']['embed'].shape, num_tokens)

            torch.save(st_delta, '/'.join(files.split('/')[:-1]) + f'/delta_epoch={epoch_number}.ckpt')
            os.remove(files)


def parse_args():
    parser = argparse.ArgumentParser('Scrape', add_help=False)
    parser.add_argument('--path', help='path of folder to checkpoints',
                        type=str)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    path = args.path
    main(path)
