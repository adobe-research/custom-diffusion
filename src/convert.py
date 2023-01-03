# Copyright 2022 Adobe Research. All rights reserved.
# To view a copy of the license, visit LICENSE.md.
import os, sys
sys.path.append('stable-diffusion')
sys.path.append('./')
import argparse
import torch
from omegaconf import OmegaConf
from diffusers import StableDiffusionPipeline
from ldm.util import instantiate_from_config
from src import diffuser_training 


def load_model_from_config(config, ckpt, verbose=False):
    print(f"Loading model from {ckpt}")
    pl_sd = torch.load(ckpt, map_location="cpu")
    if "global_step" in pl_sd:
        print(f"Global Step: {pl_sd['global_step']}")
    sd = pl_sd["state_dict"]
    model = instantiate_from_config(config.model)

    token_weights = sd["cond_stage_model.transformer.text_model.embeddings.token_embedding.weight"]
    del sd["cond_stage_model.transformer.text_model.embeddings.token_embedding.weight"]
    m, u = model.load_state_dict(sd, strict=False)
    model.cond_stage_model.transformer.text_model.embeddings.token_embedding.weight.data[:token_weights.shape[0]] = token_weights
    if len(m) > 0 and verbose:
        print("missing keys:")
        print(m)
    if len(u) > 0 and verbose:
        print("unexpected keys:")
        print(u)

    model.cuda()
    model.eval()
    return model


def convert(ckpt, delta_ckpt, newtoken, sd_version, config, compvis_to_diffuser=True):
    config = OmegaConf.load(config)
    model = load_model_from_config(config, f"{ckpt}")
    mapping_compvis_to_diffuser = {}
    mapping_compvis_to_diffuser_rev = {}
    for key in list(model.state_dict().keys()):
        if 'attn2.to_k' in key or 'attn2.to_v' in key:
            diffuser_key = key.replace('model.diffusion_model.', '')
            if 'input_blocks' in key:
                i, j = [int(x) for x in key.split('.')[3:5]]
                i_, j_ = max(0, i // 3), 0 if i in [1, 4, 7] else 1
                diffuser_key = diffuser_key.replace(f'input_blocks.{i}.{j}', f'down_blocks.{i_}.attentions.{j_}')
            if 'output_blocks' in key:
                i, j = [int(x) for x in key.split('.')[3:5]]
                i_, j_ = max(0, i // 3), 0 if i % 3 == 0 else 1 if i % 3 == 1 else 2
                diffuser_key = diffuser_key.replace(f'output_blocks.{i}.{j}', f'up_blocks.{i_}.attentions.{j_}')
            diffuser_key = diffuser_key.replace('middle_block.1', 'mid_block.attentions.0')
            mapping_compvis_to_diffuser[key] = diffuser_key
            mapping_compvis_to_diffuser_rev[diffuser_key] = key

    print(mapping_compvis_to_diffuser)
    if compvis_to_diffuser:
        st = torch.load(delta_ckpt)["state_dict"]
        diffuser_st = {'unet': {}}
        if newtoken > 0:
            diffuser_st['modifier_token'] = {}
            for i in range(newtoken):
                diffuser_st['modifier_token'][f'<new{i+1}>'] = st['embed'][i].clone()
            del st['embed']
        for key in list(st.keys()):
            diffuser_st['unet'][mapping_compvis_to_diffuser[key]] = st[key]

        torch.save(diffuser_st, f'{os.path.dirname(delta_ckpt)}/delta.bin')
        pipe = StableDiffusionPipeline.from_pretrained(sd_version, torch_dtype=torch.float16).to("cuda")
        diffuser_training.load_model(pipe.text_encoder, pipe.tokenizer, pipe.unet, f'{os.path.dirname(delta_ckpt)}/delta.bin')
        pipe.save_pretrained(os.path.dirname(delta_ckpt))
    else:
        st = torch.load(delta_ckpt)
        compvis_st = {}
        compvis_st['state_dict'] = {}
        if 'modifier_token' in st:
            compvis_st['state_dict']['embed'] = []
            for _, feat in st['modifier_token'].items():
                compvis_st['state_dict']['embed'].append(feat)
            compvis_st['state_dict']['embed'] = torch.cat(compvis_st['state_dict']['embed'])
            config.model.params.cond_stage_config.target = 'src.custom_modules.FrozenCLIPEmbedderWrapper'
            config.model.params.cond_stage_config.params = {}
            config.model.params.cond_stage_config.params.modifier_token = '+'.join([f'<new{i+1}>' for i in range(newtoken)])

        for key in list(st['unet'].keys()):
            compvis_st['state_dict'][mapping_compvis_to_diffuser_rev[key]] = st['unet'][key]

        model = load_model_from_config(config, f"{ckpt}")
        if 'modifier_token' in st:
            model.cond_stage_model.transformer.text_model.embeddings.token_embedding.weight.data[-newtoken:] = compvis_st['state_dict']['embed']
            del compvis_st['state_dict']['embed']

        model.load_state_dict(compvis_st['state_dict'], strict=False)
        torch.save({'state_dict': model.state_dict()}, f'{os.path.dirname(delta_ckpt)}/model.ckpt')


def parse_args():
    parser = argparse.ArgumentParser('Checkpoint conversion given delta ckpts, currently supported for stable diffusion 1.4 only', add_help=True)
    parser.add_argument('--ckpt', help='pretrained compvis model checkpoint',
                        type=str)
    parser.add_argument('--delta_ckpt', help='delta checkpoint either of compvis or diffuser', default=None,
                        type=str)
    parser.add_argument('--newtoken', help='number of new tokens in the checkpoint', default=1,
                        type=int)
    parser.add_argument('--sd_version', default="CompVis/stable-diffusion-v1-4",
                        type=str)
    parser.add_argument('--config', default="configs/custom-diffusion/finetune.yaml",
                        type=str)
    parser.add_argument("--compvis_to_diffuser", action='store_true')
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    assert args.sd_version == "CompVis/stable-diffusion-v1-4"
    convert(args.ckpt, args.delta_ckpt, args.newtoken, args.sd_version, args.config, args.compvis_to_diffuser)