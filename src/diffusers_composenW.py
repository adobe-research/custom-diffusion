# Copyright 2022 Adobe Research. All rights reserved.
# To view a copy of the license, visit LICENSE.md.


import sys
import os
import argparse
import torch
from scipy.linalg import lu_factor, lu_solve

sys.path.append('./')
from diffusers import StableDiffusionPipeline
from src import diffusers_sample


def gdupdateWexact(K, V, Ktarget1, Vtarget1, W, device='cuda'):
    input_ = K
    output = V
    C = input_.T@input_
    d = []
    lu, piv = lu_factor(C.cpu().numpy())
    for i in range(Ktarget1.size(0)):
        sol = lu_solve((lu, piv), Ktarget1[i].reshape(-1, 1).cpu().numpy())
        d.append(torch.from_numpy(sol).to(K.device))

    d = torch.cat(d, 1).T

    e2 = d@Ktarget1.T
    e1 = (Vtarget1.T - W@Ktarget1.T)
    delta = e1@torch.linalg.inv(e2)

    Wnew = W + delta@d
    lambda_split1 = Vtarget1.size(0)

    input_ = torch.cat([Ktarget1.T, K.T], dim=1)
    output = torch.cat([Vtarget1, V], dim=0)

    loss = torch.norm((Wnew@input_).T - output, 2, dim=1)
    print(loss[:lambda_split1].mean().item(), loss[lambda_split1:].mean().item())

    return Wnew


def compose(paths, category, outpath, pretrained_model_path, regularization_prompt, prompts, save_path, device='cuda'):
    model_id = pretrained_model_path
    pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float16).to("cuda")

    layers_modified = []
    for name, param in pipe.unet.named_parameters():
        if 'attn2.to_k' in name or 'attn2.to_v' in name:
            layers_modified.append(name)

    tokenizer = pipe.tokenizer

    def get_text_embedding(prompts):
        with torch.no_grad():
            uc = []
            for text in prompts:
                tokens = tokenizer(text,
                                   truncation=True,
                                   max_length=tokenizer.model_max_length,
                                   return_length=True,
                                   return_overflowing_tokens=False,
                                   padding="do_not_pad",
                                   ).input_ids
                if 'photo of a' in text[:15]:
                    print(text)
                    uc.append(pipe.text_encoder(torch.cuda.LongTensor(tokens).reshape(1,-1))[0][:, 4:].reshape(-1, 768))
                else:
                    uc.append(pipe.text_encoder(torch.cuda.LongTensor(tokens).reshape(1,-1))[0][:, 1:].reshape(-1, 768))
        return torch.cat(uc, 0).float()

    embeds = {}
    count = 1
    model2_sts = []
    modifier_tokens = []
    modifier_token_ids = []
    categories = []
    for path1, cat1 in zip(paths.split('+'), category.split('+')):
        model2_st = torch.load(path1)
        if 'modifier_token' in model2_st:
            # composition of models with individual concept only
            key = list(model2_st['modifier_token'].keys())[0]
            _ = tokenizer.add_tokens(f'<new{count}>')
            modifier_token_ids.append(tokenizer.convert_tokens_to_ids(f'<new{count}>'))
            modifier_tokens.append(True)
            embeds[f'<new{count}>'] = model2_st['modifier_token'][key]
        else:
            modifier_tokens.append(False)

        model2_sts.append(model2_st['unet'])
        categories.append(cat1)
        count += 1

    pipe.text_encoder.resize_token_embeddings(len(tokenizer))
    token_embeds = pipe.text_encoder.get_input_embeddings().weight.data
    for (x, y) in zip(modifier_token_ids, list(embeds.keys())):
        token_embeds[x] = embeds[y]
        print(x, y, "added embeddings")

    f = open(regularization_prompt, 'r')
    prompt = [x.strip() for x in f.readlines()][:200]
    uc = get_text_embedding(prompt)

    uc_targets = []
    from collections import defaultdict
    uc_values = defaultdict(list)
    for composing_model_count in range(len(model2_sts)):
        category = categories[composing_model_count]
        if modifier_tokens[composing_model_count]:
            string1 = f'<new{composing_model_count+1}> {category}'
        else:
            string1 = f'{category}'
        if 'art' in string1:
            prompt = [string1] + [f"painting in the style of {string1}"]
        else:
            prompt = [string1] + [f"photo of a {string1}"]
        uc_targets.append(get_text_embedding(prompt))
        for each in layers_modified:
            uc_values[each].append((model2_sts[composing_model_count][each].to(device)@uc_targets[-1].T).T)

    uc_targets = torch.cat(uc_targets, 0)

    removal_indices = []
    for i in range(uc_targets.size(0)):
        for j in range(i+1, uc_targets.size(0)):
            if (uc_targets[i]-uc_targets[j]).abs().mean() == 0:
                removal_indices.append(j)

    removal_indices = list(set(removal_indices))
    uc_targets = torch.stack([uc_targets[i] for i in range(uc_targets.size(0)) if i not in removal_indices], 0)
    for each in layers_modified:
        uc_values[each] = torch.cat(uc_values[each], 0)
        uc_values[each] = torch.stack([uc_values[each][i] for i in range(uc_values[each].size(0)) if i not in removal_indices], 0)
        print(uc_values[each].size(), each)

    print("target size:", uc_targets.size())

    new_weights = {'unet': {}}
    for each in layers_modified:
        W = pipe.unet.state_dict()[each].float()
        values = (W@uc.T).T
        input_target = uc_targets
        output_target = uc_values[each]

        Wnew = gdupdateWexact(uc[:values.shape[0]],
                              values,
                              input_target,
                              output_target,
                              W.clone(),
                              )

        new_weights['unet'][each] = Wnew
        print(Wnew.size())

    new_weights['modifier_token'] = embeds
    os.makedirs(f'{save_path}/{outpath}', exist_ok=True)
    torch.save(new_weights, f'{save_path}/{outpath}/delta.bin')

    if prompts is not None:
        if os.path.exists(prompts):
            diffusers_sample.sample(model_id, f'{save_path}/{outpath}/delta.bin', prompts, prompt=None, compress=False, freeze_model='crossattn_kv', batch_size=1)
        else:
            diffusers_sample.sample(model_id, f'{save_path}/{outpath}/delta.bin', from_file=None, prompt=prompts, compress=False, freeze_model='crossattn_kv', batch_size=1)


def parse_args():
    parser = argparse.ArgumentParser('', add_help=False)
    parser.add_argument('--paths', help='+ separated list of checkpoints', required=True,
                        type=str)
    parser.add_argument('--save_path', help='folder name to save  optimized weights', default='optimized_logs',
                        type=str)
    parser.add_argument('--categories', help='+ separated list of categories of the models', required=True,
                        type=str)
    parser.add_argument('--prompts', help='prompts for composition model (can be a file or string)', default=None,
                        type=str)
    parser.add_argument('--ckpt', required=True,
                        type=str)
    parser.add_argument('--regularization_prompt', default='./data/regularization_captions.txt',
                        type=str)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    paths = args.paths
    categories = args.categories
    if ' ' in categories:
        temp = categories.replace(' ', '_')
    else:
        temp = categories
    outpath = '_'.join(['optimized', temp])
    compose(paths, categories, outpath, args.ckpt, args.regularization_prompt, args.prompts, args.save_path)
