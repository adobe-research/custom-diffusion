# Copyright 2022 Adobe Research. All rights reserved.
# To view a copy of the license, visit LICENSE.md.
import os
import argparse
import glob
import torch


def main(path, newtoken=0):
    layers = []
    for files in glob.glob(f'{path}/checkpoints/*'):
        if ('=' in files or '_' in files) and 'delta' not in files:
            print(files)
            if '=' in files:
                epoch_number = files.split('=')[1].split('.ckpt')[0]
            elif '_' in files:
                epoch_number = files.split('/')[-1].split('.ckpt')[0]

            st = torch.load(files)["state_dict"]
            if len(layers) == 0:
                for key in list(st.keys()):
                    if 'attn2.to_k' in key or 'attn2.to_v' in key:
                        layers.append(key)
                print(layers)
            st_delta = {'state_dict': {}}
            for each in layers:
                st_delta['state_dict'][each] = st[each].clone()
            print('/'.join(files.split('/')[:-1]) + f'/delta_epoch={epoch_number}.ckpt')

            num_tokens = st['cond_stage_model.transformer.text_model.embeddings.token_embedding.weight'].shape[0]

            if newtoken > 0:
                print("saving the optimized embedding")
                st_delta['state_dict']['embed'] = st['cond_stage_model.transformer.text_model.embeddings.token_embedding.weight'][-newtoken:].clone()
                print(st_delta['state_dict']['embed'].shape, num_tokens)

            torch.save(st_delta, '/'.join(files.split('/')[:-1]) + f'/delta_epoch={epoch_number}.ckpt')
            os.remove(files)


def parse_args():
    parser = argparse.ArgumentParser('', add_help=False)
    parser.add_argument('--path', help='path of folder to checkpoints',
                        type=str)
    parser.add_argument('--newtoken', help='number of new tokens in the checkpoint', default=1,
                        type=int)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    path = args.path
    main(path, args.newtoken)
