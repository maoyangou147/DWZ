from functools import partial

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .mmseg.models.sam import ImageEncoderViT_TS, FeatureFusionDecoder

class SAM(nn.Module):
    def __init__(self):
        super().__init__()
        self.image_encoder = ImageEncoderViT_TS(
            img_size=1024,
            patch_size=16,
            in_chans=3,
            embed_dim=768,
            depth=12,
            num_heads=12,
            mlp_ratio=4,
            out_chans=256,
            qkv_bias=True,
            norm_layer=partial(torch.nn.LayerNorm, eps=1e-6),
            act_layer=nn.GELU,
            use_rel_pos=True,
            rel_pos_zero_init=True,
            window_size=14,
            global_attn_indexes=(2,5,8,11),
            use_fft=False,
            scale_factor=3,
            csa_block_indice=(10,11),     
            mrm_block_indice=(10,11)
        )
        self.mask_decoder = FeatureFusionDecoder()
        self.width_list = [256, 512, 1024]

        for name, para in self.image_encoder.named_parameters():
            if "csa_embedding" in name:
                continue
            elif "csa_blocks" in name:
                continue  
            else:
                para.requires_grad_(False)


    def forward(self, x):
        features = self.image_encoder(x)
        output = self.mask_decoder(features)
        return output


