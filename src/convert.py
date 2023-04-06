# Copyright 2022 Adobe Research. All rights reserved.
# To view a copy of the license, visit LICENSE.md.
import os, sys
import argparse
import torch
from omegaconf import OmegaConf
from ldm.util import instantiate_from_config

sys.path.append('stable-diffusion')
sys.path.append('./')
from src.diffusers_model_pipeline import CustomDiffusionPipeline


def load_model_from_config(config, ckpt, verbose=False):
    print(f"Loading model from {ckpt}")
    pl_sd = torch.load(ckpt, map_location="cpu")
    if "global_step" in pl_sd:
        print(f"Global Step: {pl_sd['global_step']}")
    sd = pl_sd["state_dict"]
    model = instantiate_from_config(config.model)
    m, u = model.load_state_dict(sd, strict=False)
    model.cuda()
    model.eval()
    return model


def load_model_from_config_addtoken(config, ckpt, verbose=False):
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


def convert(ckpt, delta_ckpt, sd_version, config, modelname, mode):
    config = OmegaConf.load(config)
    model = load_model_from_config(config, f"{ckpt}")
    # get the mapping of layer names between diffuser and CompVis checkpoints
    mapping_compvis_to_diffuser = {}
    mapping_compvis_to_diffuser_rev = {}
    for key in list(model.state_dict().keys()):
        if 'attn2' in key:
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

    # convert checkpoint to webui
    if mode in ['diffuser-to-webui' or 'compvis-to-webui']:
        outpath = f'{os.path.dirname(delta_ckpt)}/webui'
        os.makedirs(outpath, exist_ok=True)
        if mode == 'diffuser-to-webui':
            st = torch.load(delta_ckpt)
            compvis_st = {}
            compvis_st['state_dict'] = {}
            for key in list(st['unet'].keys()):
                compvis_st['state_dict'][mapping_compvis_to_diffuser_rev[key]] = st['unet'][key]

            model.load_state_dict(compvis_st['state_dict'], strict=False)
            torch.save({'state_dict': model.state_dict()}, f'{outpath}/{modelname}')

            if 'modifier_token' in st:
                os.makedirs(f'{outpath}/embeddings/', exist_ok=True)
                for word, feat in st['modifier_token'].items():
                    torch.save({word: feat}, f'{outpath}/embeddings/{word}.pt')
        else:
            compvis_st = torch.load(delta_ckpt)["state_dict"]
            model.load_state_dict(compvis_st['state_dict'], strict=False)
            torch.save({'state_dict': model.state_dict()}, f'{outpath}/{modelname}')

            if 'embed' in st:
                os.makedirs(f'{outpath}/embeddings/', exist_ok=True)
                for i, feat in enumerate(st['embed']):
                    torch.save({f'<new{i}>': feat}, f'{outpath}/embeddings/<new{i}>.pt')
    # convert checkpoint from CompVis to diffuser
    elif mode == 'compvis-to-diffuser':
        st = torch.load(delta_ckpt)["state_dict"]
        diffuser_st = {'unet': {}}
        if 'embed' in st:
            diffuser_st['modifier_token'] = {}
            for i in range(st['embed'].size(0)):
                diffuser_st['modifier_token'][f'<new{i+1}>'] = st['embed'][i].clone()
            del st['embed']
        for key in list(st.keys()):
            diffuser_st['unet'][mapping_compvis_to_diffuser[key]] = st[key]
        torch.save(diffuser_st, f'{os.path.dirname(delta_ckpt)}/delta.bin')
        pipe = CustomDiffusionPipeline.from_pretrained(sd_version, torch_dtype=torch.float16).to("cuda")
        pipe.load_model(f'{os.path.dirname(delta_ckpt)}/delta.bin')
        pipe.save_pretrained(os.path.dirname(delta_ckpt), all=True)
    # convert checkpoint from diffuser to CompVis
    elif mode == 'diffuser-to-compvis':
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
            config.model.params.cond_stage_config.params.modifier_token = '+'.join([f'<new{i+1}>' for i in range(len(st['modifier_token']))])

        for key in list(st['unet'].keys()):
            compvis_st['state_dict'][mapping_compvis_to_diffuser_rev[key]] = st['unet'][key]

        torch.save(compvis_st, f'{os.path.dirname(delta_ckpt)}/delta_model.ckpt')
        model = load_model_from_config_addtoken(config, f"{ckpt}")
        if 'modifier_token' in st:
            model.cond_stage_model.transformer.text_model.embeddings.token_embedding.weight.data[-len(st['modifier_token']):] = compvis_st['state_dict']['embed']
            del compvis_st['state_dict']['embed']

        model.load_state_dict(compvis_st['state_dict'], strict=False)
        torch.save({'state_dict': model.state_dict()}, f'{os.path.dirname(delta_ckpt)}/model.ckpt')


def parse_args():
    parser = argparse.ArgumentParser('Checkpoint conversion given delta ckpts, currently supported for stable diffusion 1.4 only', add_help=True)
    parser.add_argument('--ckpt', help='pretrained compvis model checkpoint', required=True,
                        type=str)
    parser.add_argument('--delta_ckpt', help='delta checkpoint either of compvis or diffuser', required=True,
                        type=str)
    parser.add_argument('--sd_version', default="CompVis/stable-diffusion-v1-4",
                        type=str)
    parser.add_argument('--config', default="configs/custom-diffusion/finetune.yaml",
                        type=str)
    parser.add_argument('--modelname', default="model.ckpt", help="name of the model to save when converting to webui",
                        type=str)
    parser.add_argument("--mode", default='compvis-to-diffuser', choices=['diffuser-to-webui', 'compvis-to-webui', 'compvis-to-diffuser', 'diffuser-to-compvis'],
                        type=str)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    assert args.sd_version == "CompVis/stable-diffusion-v1-4"
    convert(args.ckpt, args.delta_ckpt, args.sd_version, args.config, args.modelname, args.mode)
