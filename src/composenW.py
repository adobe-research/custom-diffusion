# Copyright 2022 Adobe Research. All rights reserved.
# To view a copy of the license, visit LICENSE.md.


import sys
import os
import argparse
import random
import torch
import torchvision

import numpy as np
from tqdm import tqdm
from scipy.linalg import lu_factor, lu_solve

sys.path.append('stable-diffusion')
sys.path.append('./')
from omegaconf import OmegaConf
from ldm.util import instantiate_from_config
from ldm.models.diffusion.ddim import DDIMSampler


def load_model_from_config(config, ckpt):
    print(f"Loading model from {ckpt}")
    pl_sd = torch.load(ckpt, map_location="cpu")
    sd = pl_sd["state_dict"]
    model = instantiate_from_config(config.model)
    m, u = model.load_state_dict(sd, strict=False)
    model.cuda()
    model.eval()
    return model


def get_model(path):
    config = OmegaConf.load("configs/custom-diffusion/finetune.yaml")
    model = load_model_from_config(config, path)
    return model, config


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
    model, config = get_model(pretrained_model_path)
    model.eval()
    model.requires_grad = False

    layers = []
    layers_modified = []

    def getlayers(model, root_name=''):
        for name, module in model.named_children():
            if module.__class__.__name__ == 'SpatialTransformer':
                layers_modified.append(root_name + '.' + name + '.transformer_blocks.0.attn2.to_k')
                layers_modified.append(root_name + '.' + name + '.transformer_blocks.0.attn2.to_v')
            else:
                if list(module.children()) == []:
                    layers.append(root_name + '.' + name)
                else:
                    getlayers(module, root_name + '.' + name)

    getlayers(model.model.diffusion_model)

    for i in range(len(layers_modified)):
        layers_modified[i] = 'model.diffusion_model' + layers_modified[i] + '.weight'

    def get_text_embedding(prompts):
        with torch.no_grad():
            uc = []
            for text in prompts:
                tokens = tokenizer(text,
                                   truncation=True,
                                   max_length=77,
                                   return_length=True,
                                   return_overflowing_tokens=False,
                                   padding="max_length",
                                   return_tensors="pt")

                tokens = tokens["input_ids"]
                end = torch.nonzero(tokens == 49407)[:, 1].min()
                if 'photo of a' in text[:15]:
                    print(text)
                    uc.append((model.get_learned_conditioning(1 * [text])[:, 4:end+1]).reshape(-1, 768))
                else:
                    uc.append((model.get_learned_conditioning(1 * [text])[:, 1:end+1]).reshape(-1, 768))

        return torch.cat(uc, 0)

    tokenizer = model.cond_stage_model.tokenizer
    embeds = []
    count = 1

    model2_sts = []
    modifier_tokens = []
    categories = []
    config.model.params.cond_stage_config.params = {}
    config.model.params.cond_stage_config.params.modifier_token = None
    for path1, cat1 in zip(paths.split('+'), category.split('+')):
        model2_st = torch.load(path1)
        if 'embed' in model2_st['state_dict']:
            config.model.params.cond_stage_config.target = 'src.custom_modules.FrozenCLIPEmbedderWrapper'
            embeds.append(model2_st['state_dict']['embed'][-1:])
            num_added_tokens1 = tokenizer.add_tokens(f'<new{count}>')
            modifier_token_id1 = tokenizer.convert_tokens_to_ids('<new1>')
            modifier_tokens.append(True)
            if config.model.params.cond_stage_config.params.modifier_token is None:
                config.model.params.cond_stage_config.params.modifier_token = f'<new{count}>'
            else:
                config.model.params.cond_stage_config.params.modifier_token += f'+<new{count}>'
        else:
            modifier_tokens.append(False)

        model2_sts.append(model2_st['state_dict'])
        categories.append(cat1)
        count += 1

    embeds = torch.cat(embeds, 0)
    model.cond_stage_model.transformer.resize_token_embeddings(len(tokenizer))
    token_embeds = model.cond_stage_model.transformer.get_input_embeddings().weight.data
    token_embeds[-embeds.size(0):] = embeds

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
            prompt = [string1] + [f"a photo of {string1}"]
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

    new_weights = {}
    for each in layers_modified:
        values = (model.state_dict()[each]@uc.T).T
        input_target = uc_targets
        output_target = uc_values[each]

        Wnew = gdupdateWexact(uc[:values.shape[0]],
                              values,
                              input_target,
                              output_target,
                              model.state_dict()[each].clone(),
                              )

        new_weights[each] = Wnew
        print(Wnew.size())

    if prompts is not None:
        model.load_state_dict(new_weights, strict=False)
        sampler = DDIMSampler(model)
        sampler.make_schedule(ddim_num_steps=200, ddim_eta=1., verbose=False)

        seed = 68
        os.environ['PYTHONHASHSEED'] = str(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        np.random.seed(seed)
        random.seed(seed)

        batch_size = 10

        if not os.path.exists(prompts):
            assert prompts is not None
            prompts = [batch_size * [prompts]]

        else:
            print(f"reading prompts from {prompts}")
            with open(prompts, "r") as f:
                prompts = f.read().splitlines()
                prompts = [batch_size * [prompt] for prompt in prompts]
                print(prompts[0])

        sample_path = os.path.join(f'{save_path}/{outpath}/', 'samples')
        os.makedirs(sample_path, exist_ok=True)
        with torch.no_grad():
            for counter, prompt in enumerate(prompts):
                print(prompt)
                uc_try = model.get_learned_conditioning(batch_size * [prompt[0]])

                unconditional_guidance_scale = 6.
                cond = uc_try
                unconditional_conditioning = model.get_learned_conditioning(batch_size * [""])

                img = torch.randn((batch_size, 4, 64, 64)).cuda()
                ddim_use_original_steps = False

                timesteps = sampler.ddpm_num_timesteps if ddim_use_original_steps else sampler.ddim_timesteps
                time_range = reversed(range(0, timesteps)) if ddim_use_original_steps else np.flip(timesteps)
                total_steps = timesteps if ddim_use_original_steps else timesteps.shape[0]
                iterator = tqdm(time_range, desc='DDIM Sampler', total=total_steps)

                for i, step in enumerate(iterator):
                    index = total_steps - i - 1
                    ts = torch.full((batch_size,), step, device=device, dtype=torch.long)
                    outs = sampler.p_sample_ddim(img, cond, ts, index=index, use_original_steps=ddim_use_original_steps,
                                                 unconditional_guidance_scale=unconditional_guidance_scale,
                                                 unconditional_conditioning=unconditional_conditioning)
                    img, _ = outs

                outim = model.decode_first_stage(outs[0])
                outim = torch.clamp((outim + 1.0) / 2.0, min=0.0, max=1.0)
                name = '-'.join(prompt[0].split(' '))
                torchvision.utils.save_image(outim, f'{save_path}/{outpath}/{counter}_{name}.jpg', nrow=batch_size // 2)

    new_weights['embed'] = embeds
    os.makedirs(f'{save_path}/{outpath}', exist_ok=True)
    os.makedirs(f'{save_path}/{outpath}/checkpoints', exist_ok=True)
    os.makedirs(f'{save_path}/{outpath}/configs', exist_ok=True)
    with open(f'{save_path}/{outpath}/configs/config_project.yaml', 'w') as fp:
        OmegaConf.save(config=config, f=fp)
    torch.save({'state_dict': new_weights}, f'{save_path}/{outpath}/checkpoints/delta_epoch=000000.ckpt')


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
