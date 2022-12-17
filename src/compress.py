# Copyright 2022 Adobe Research. All rights reserved.
# To view a copy of the license, visit LICENSE.md.

import os
import torch
import argparse


def compress(delta_ckpt, ckpt, device='cuda'):
    st = torch.load(f'{delta_ckpt}')
    pretrained_st = torch.load(ckpt)['state_dict']
    for each in pretrained_st.keys():
        if 'attn2' in each:
            print(each)

    embed = None
    if 'embed' in st['state_dict']:
        embed = st['state_dict']['embed']
        del st['state_dict']['embed']

    compressed_st = {'state_dict': {}}

    layers = list(st['state_dict'].keys())
    print("getting compression")

    for name in layers:
        # print(each)
        W = st['state_dict'][name].to(device)
        Wpretrain = pretrained_st[name].clone().to(device)
        deltaW = W-Wpretrain

        u, s, vt = torch.linalg.svd(deltaW.clone())

        explain = 0 
        all_ = (s).sum()
        for i, t in enumerate(s):
            explain += t/(all_)
            if explain > .6:
                break
        
        compressed_st['state_dict'][f'{name}'] = {}
        compressed_st['state_dict'][f'{name}']['u'] = (u[:, :i]@torch.diag(s)[:i, :i]).clone()
        compressed_st['state_dict'][f'{name}']['v'] = vt[:i].clone()

    if embed is not None:
        compressed_st['state_dict']['embed'] = embed.clone()

    name = delta_ckpt.replace('delta', 'compressed_delta')
    torch.save(compressed_st, f'{name}')


def parse_args():
    parser = argparse.ArgumentParser('', add_help=False)
    parser.add_argument('--delta_ckpt', help='path of checkpoint to compress',
                        type=str)
    parser.add_argument('--ckpt', help='path of pretrained model checkpoint',
                        type=str)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    compress(args.delta_ckpt, args.ckpt)
