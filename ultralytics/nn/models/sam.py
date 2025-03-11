from functools import partial

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .mmseg.models.sam import ImageEncoderViT_TS, FeatureAdapter, LoraViT


class LoRA(nn.Module):
    def __init__(self, base_layer: nn.Linear, r: int = 8, alpha: float = 1.0):
        super().__init__()
        self.base_layer = base_layer
        self.r = r
        self.alpha = alpha

        d_in, d_out = base_layer.in_features, base_layer.out_features
        self.A = nn.Parameter(torch.randn(d_in, r))
        self.B = nn.Parameter(torch.zeros(r, d_out))
        nn.init.kaiming_normal_(self.A, mode='fan_in', nonlinearity='linear')

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        base_output = self.base_layer(x)
        lora_output = (x @ self.A @ self.B) * self.alpha
        # print(
        #     f"LoRA 项统计: max={lora_output.abs().max().item():.3e}, "
        #     f"mean={lora_output.mean().item():.3e}, std={lora_output.std().item():.3e}"
        # )

        return base_output + lora_output


'''
class LoRA(nn.Module):
    def __init__(self, base_layer, r=8, alpha=1.0):
        super().__init__()
        self.base = base_layer
        self.A = nn.Parameter(torch.randn(base_layer.in_features, r) * 0.02)  # 缩小初始化
        self.B = nn.Parameter(torch.zeros(r, base_layer.out_features))
        self.alpha = alpha / r  # 动态缩放因子
        
    def forward(self, x):
        # 输入规范化
        x = x.to(torch.float32)  # 强制FP32计算
        x = x / (x.norm(dim=1, keepdim=True) + 1e-6)  # L2规范化
        
        # 安全矩阵乘法
        lora_output = (x @ self.A.clamp(-5,5) @ self.B.clamp(-5,5)) * self.alpha
        lora_output = lora_output.clamp(-10, 10)  # 输出限幅
        
        print(
            f"LORA 项统计:max={lora_output.abs().max().item():.3e},"
            f"mean={lora_output.mean().item():.3e}, std={lora_output.std().item():.3e}"
        )
        
        return self.base(x) + lora_output.to(x.dtype)
'''


def apply_lora(
    model: nn.Module, 
    r: int = 8, 
    alpha: float = 1.0, 
    target_layers: list = ["11.attn.qkv", "11.attn.proj"]
    # target_layers: list = ["11.attn.qkv"]
) -> nn.Module:
    """
    将模型中的目标线性层替换为 LoRA 模块
    """
    for name, module in model.named_modules():
        # 检查是否是目标层（注意力层的 qkv 和 proj 层）
        if any([layer_name in name for layer_name in target_layers]) and isinstance(module, nn.Linear):
            # 替换为 LoRA 模块
            parent = model
            parts = name.split('.')
            for part in parts[:-1]:
                parent = getattr(parent, part)
            original_layer = getattr(parent, parts[-1])
            setattr(parent, parts[-1], LoRA(original_layer, r=r, alpha=alpha))
    return model


class SAM(nn.Module):
    def __init__(self):
        super().__init__()
        vit_model = LoraViT(
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
            global_attn_indexes=(2,5,8,11)
        )

        sam_checkpoint = torch.load('/root/ckpt/sam_vit_b_01ec64.pth')
        sam_checkpoint = {k.replace("image_encoder.", "", 1): v for k, v in sam_checkpoint.items() if 'image_encoder' in k}
        # for name, param in sam_checkpoint.items():
        #     print(f"Parameter name: {name}, Shape: {tuple(param.shape)}")
        vit_model.load_state_dict(sam_checkpoint)

        self.image_encoder = apply_lora(vit_model, r=8, alpha=1.0)


        self.mask_decoder = FeatureAdapter(c1=64, c2=128, c3=256)   # yolo v8n
        # self.mask_decoder = FeatureAdapter(c1=128, c2=256, c3=512)   # yolo v8s
        # self.mask_decoder = FeatureAdapter(c1=192, c2=384, c3=576)   # yolo v8m
        # self.mask_decoder = FeatureAdapter(c1=256, c2=512, c3=512)   # yolo v8l
        # self.mask_decoder = FeatureAdapter(c1=320, c2=640, c3=640)   # yolo v8x
        self.width_list = [64, 128, 256]   # yolo v8n
        # self.width_list = [128, 256, 512]   # yolo v8s
        # self.width_list = [192, 384, 576]   # yolo v8m
        # self.width_list = [256, 512, 512]   # yolo v8l
        # self.width_list = [320, 640, 640]   # yolo v8x

        for name, para in self.image_encoder.named_parameters():
            if "A" in name or "B" in name:
                para.requires_grad_(True)  
            else:
                para.requires_grad_(False)


    def forward(self, x):
        features = self.image_encoder(x)
        output = self.mask_decoder(features)
        return output
        # feat_1, feat_2, feat_3 = self.features


