import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint

import open_clip

class AbstractEncoder(nn.Module):
    def __init__(self):
        super().__init__()

    def encode(self, *args, **kwargs):
        raise NotImplementedError


class FrozenOpenCLIPEmbedderWrapper(AbstractEncoder):
    """
    Uses the OpenCLIP transformer encoder for text
    """
    LAYERS = [
        #"pooled",
        "last",
        "penultimate"
    ]
    def __init__(self, arch="ViT-H-14", version="laion2b_s32b_b79k", device="cuda", max_length=77,
                 freeze=True, layer="last"):
        super().__init__()
        assert layer in self.LAYERS
        model, _, _ = open_clip.create_model_and_transforms(arch, device=torch.device('cpu'), pretrained=version)
        del model.visual
        self.model = model

        self.device = device
        self.max_length = max_length
        if freeze:
            self.freeze()
        self.layer = layer
        if self.layer == "last":
            self.layer_idx = 0
        elif self.layer == "penultimate":
            self.layer_idx = 1
        else:
            raise NotImplementedError()

    def freeze(self):
        self.model = self.model.eval()
        for param in self.parameters():
            param.requires_grad = False

    def forward(self, text):
        tokens = open_clip.tokenize(text)
        z = self.encode_with_transformer(tokens.to(self.device))
        return z

    def encode_with_transformer(self, text):
        x = self.model.token_embedding(text)  # [batch_size, n_ctx, d_model]
        x = x + self.model.positional_embedding
        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.text_transformer_forward(x, attn_mask=self.model.attn_mask)
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.model.ln_final(x)
        return x

    def text_transformer_forward(self, x: torch.Tensor, attn_mask = None):
        for i, r in enumerate(self.model.transformer.resblocks):
            if i == len(self.model.transformer.resblocks) - self.layer_idx:
                break
            if self.model.transformer.grad_checkpointing and not torch.jit.is_scripting():
                x = checkpoint(r, x, attn_mask)
            else:
                x = r(x, attn_mask=attn_mask)
        return x

    def encode(self, text):
        return self(text)


# class FrozenOpenCLIPEmbedderWrapper(AbstractEncoder):
#     """
#     Uses the OpenCLIP transformer encoder for text
#     """
#     LAYERS = [
#         #"pooled",
#         "last",
#         "penultimate"
#     ]
#     def __init__(self, modifier_token, initializer_token, arch="ViT-H-14", version="laion2b_s32b_b79k", device="cuda", max_length=77,
#                  freeze=True, layer="last"):
#         super().__init__()
#         assert layer in self.LAYERS
#         model, _, _ = open_clip.create_model_and_transforms(arch, device=torch.device('cpu'), pretrained=version)
#         self.tokenizer = open_clip.get_tokenizer(arch)
#         del model.visual
#         self.model = model

#         self.device = device
#         self.max_length = max_length

#         self.layer = layer
#         if self.layer == "last":
#             self.layer_idx = 0
#         elif self.layer == "penultimate":
#             self.layer_idx = 1
#         else:
#             raise NotImplementedError()

#         self.modifier_token = modifier_token
#         self.initializer_token = initializer_token
#         if '+' in self.modifier_token:
#             self.modifier_token = self.modifier_token.split('+')
#             self.initializer_token = self.initializer_token.split('+')
#         else:
#             self.modifier_token = [self.modifier_token]
#             self.initializer_token = [self.initializer_token]

#         self.add_token()
#         if freeze:
#             self.freeze()

#     def add_token(self):
#         self.modifier_token_id = []
#         token_embeds1 = self.model.token_embedding.weight.data
#         self.model.token_embedding = nn.Embedding((token_embeds1.size(0)+1, token_embeds.size(1)))
#         self.model.token_embedding.weight.data[:-1, :] = token_embeds1

#         for each_modifier_token, each_initializer_token in zip(self.modifier_token, self.initializer_token):
#             tokens = self.tokenizer(each_initializer_token)
#             num_added_tokens = self.tokenizer.add_tokens(each_modifier_token)
#             modifier_token_id = self.tokenizer.convert_tokens_to_ids(each_modifier_token)
#             self.modifier_token_id.append(modifier_token_id)
#             print(tokens)
#             self.model.token_embedding.weight.data[modifier_token_id, :] = token_embeds1[tokens[0]]

#     def freeze(self):
#         self.model = self.model.eval()
#         for param in self.parameters():
#             param.requires_grad = False

#     def forward(self, text):
#         tokens = self.tokenizer(text)
#         z = self.encode_with_transformer(tokens.to(self.device))
#         return z

#     def encode_with_transformer(self, text):
#         x = self.model.token_embedding(text)  # [batch_size, n_ctx, d_model]
#         x = x + self.model.positional_embedding
#         x = x.permute(1, 0, 2)  # NLD -> LND
#         x = self.text_transformer_forward(x, attn_mask=self.model.attn_mask)
#         x = x.permute(1, 0, 2)  # LND -> NLD
#         x = self.model.ln_final(x)
#         return x

#     def text_transformer_forward(self, x: torch.Tensor, attn_mask = None):
#         for i, r in enumerate(self.model.transformer.resblocks):
#             if i == len(self.model.transformer.resblocks) - self.layer_idx:
#                 break
#             if self.model.transformer.grad_checkpointing and not torch.jit.is_scripting():
#                 x = checkpoint(r, x, attn_mask)
#             else:
#                 x = r(x, attn_mask=attn_mask)
#         return x

#     def encode(self, text):
#         return self(text)
