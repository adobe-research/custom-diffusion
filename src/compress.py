# Copyright 2022 Adobe Research. All rights reserved.
# To view a copy of the license, visit LICENSE.md.

import torch
import argparse


def compress(delta_ckpt, ckpt, diffuser=False, compression_ratio=0.6, device='cuda'):
    st = torch.load(f'{delta_ckpt}')

    if not diffuser:
        compressed_key = 'state_dict'
        compressed_st = {compressed_key: {}}
        pretrained_st = torch.load(ckpt)['state_dict']
        if 'embed' in st['state_dict']:
            compressed_st['state_dict']['embed'] = st['state_dict']['embed']
            del st['state_dict']['embed']

        st = st['state_dict']
    else:
        from diffusers import StableDiffusionPipeline
        compressed_key = 'unet'
        compressed_st = {compressed_key: {}}
        pretrained_st = StableDiffusionPipeline.from_pretrained(ckpt, torch_dtype=torch.float16).to("cuda")
        pretrained_st = pretrained_st.unet.state_dict()
        if 'modifier_token' in st:
            compressed_st['modifier_token'] = st['modifier_token']
        st = st['unet']

    print("getting compression")
    layers = list(st.keys())
    for name in layers:
        if 'to_k' in name or 'to_v' in name:
            W = st[name].to(device)
            Wpretrain = pretrained_st[name].clone().to(device)
            deltaW = W-Wpretrain

            u, s, vt = torch.linalg.svd(deltaW.clone())

            explain = 0
            all_ = (s).sum()
            for i, t in enumerate(s):
                explain += t/(all_)
                if explain > compression_ratio:
                    break

            compressed_st[compressed_key][f'{name}'] = {}
            compressed_st[compressed_key][f'{name}']['u'] = (u[:, :i]@torch.diag(s)[:i, :i]).clone()
            compressed_st[compressed_key][f'{name}']['v'] = vt[:i].clone()
        else:
            compressed_st[compressed_key][f'{name}'] = st[name]

    name = delta_ckpt.replace('delta', 'compressed_delta')
    torch.save(compressed_st, f'{name}')


def parse_args():
    parser = argparse.ArgumentParser('', add_help=False)
    parser.add_argument('--delta_ckpt', help='path of checkpoint to compress',
                        type=str)
    parser.add_argument('--ckpt', help='path of pretrained model checkpoint',
                        type=str)
    parser.add_argument("--diffuser", action='store_true')
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    compress(args.delta_ckpt, args.ckpt, args.diffuser)
