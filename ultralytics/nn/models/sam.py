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
            csa_block_indice=(0,1,2,3,4,5,6,7,8,9,10,11),     
            mrm_block_indice=(0,1,2,3,4,5,6,7,8,9,10,11)
        )
        self.mask_decoder = FeatureFusionDecoder(in_dims=256)

        for name, para in self.image_encoder.named_parameters():
            if "csa_embedding" in name:
                continue
            elif "mrm_embedding" in name:
                continue
            elif "csa_blocks" in name:
                continue  
            elif "mrm_blocks" in name:
                continue
            elif "fft_embedding" in name:
                continue
            else:
                para.requires_grad_(False)


    def forward(self, x):
        features = self.image_encoder(x)
        output = self.mask_decoder(features)
        return output
        # feat_1, feat_2, feat_3 = self.features


    def set_requires_grad(self, nets, requires_grad=False):
        """Set requies_grad=Fasle for all the networks to avoid unnecessary computations
        Parameters:
            nets (network list)   -- a list of networks
            requires_grad (bool)  -- whether the networks require gradients or not
        """
        if not isinstance(nets, list):
            nets = [nets]
        for net in nets:
            if net is not None:
                for param in net.parameters():
                    param.requires_grad = requires_grad
