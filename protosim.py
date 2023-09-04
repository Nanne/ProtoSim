import sys
sys.path.insert(0, './dino')
from functools import partial
import math
import numpy as np

import torch.nn as nn
import torch
import torch.nn.functional as F

from vision_transformer import VisionTransformer, Block
from torchvision.models.resnet import ResNet, BasicBlock, Bottleneck

class ProtosimResNet(ResNet):
    """ Protosim with ResNet - not really tested """
    def __init__(self, num_prototypes=128, prototype_dim=256, block = BasicBlock,
        layers = [],
        num_classes = 1000,
        zero_init_residual = False,
        groups = 1,
        width_per_group = 64,
        replace_stride_with_dilation = None,
        norm_layer = None,
        **kwargs):

        super().__init__(block=block, layers=layers, num_classes=num_classes, zero_init_residual=zero_init_residual, groups=groups, width_per_group=width_per_group,
                replace_stride_with_dilation=replace_stride_with_dilation, norm_layer=norm_layer, **kwargs)


        #for p in self.parameters():
        #    p.requires_grad = False

        self.embed_dim = self.fc.weight.shape[1]

        self.protoAT = ProtoAttention(num_prototypes=num_prototypes, dim=self.embed_dim)

        self.final_block = Block(
                dim=self.embed_dim, num_heads=1, qkv_bias=True)

        #self.blocks[-1] = self.blocks[-1].attn
        #self.blocks[-1].proj = None


    def features(self, x):
        x = self.prepare_tokens(x)
        for blk in self.blocks:
            x = blk.forward(x)
        x = self.norm(x)
        #x, attn = self.blocks[-1](x, return_heads=True)
        return x

    def forward(self, x, return_attn=False, return_final=False):
        x = self.features(x)

        x = self.protoAT(x, return_attn=return_attn)
        if return_attn:
            return x
        x = self.final_block(x, return_attention=return_final)
        if return_final:
            return x
        x = self.norm(x)
        return x[:,0,:]

class ProtosimTransformer(VisionTransformer):
    """ Protosim Transformer """
    def __init__(self, num_prototypes=128, prototype_dim=256, img_size=[224], patch_size=16, in_chans=3, num_classes=0, embed_dim=768, depth=12,
                 num_heads=12, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop_rate=0., attn_drop_rate=0.,
                 drop_path_rate=0., norm_layer=nn.LayerNorm, **kwargs):

        super().__init__(img_size=img_size, patch_size=patch_size, in_chans=in_chans, num_classes=num_classes, embed_dim=embed_dim, depth=depth,
                 num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale, drop_rate=drop_rate, attn_drop_rate=attn_drop_rate,
                 drop_path_rate=drop_path_rate, norm_layer=norm_layer, **kwargs)


        for p in self.parameters():
            p.requires_grad = False

        self.protoAT = ProtoAttention(num_prototypes=num_prototypes, dim=embed_dim)

        self.final_block = Block(
                dim=embed_dim, num_heads=1, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                drop=drop_rate, attn_drop=attn_drop_rate, drop_path=self.blocks[-1].drop_path.drop_prob if drop_path_rate > 0. else 0., norm_layer=norm_layer)

    def features(self, x):
        x = self.prepare_tokens(x)
        for blk in self.blocks:
            x = blk.forward(x)
        x = self.norm(x)
        #x, attn = self.blocks[-1](x, return_heads=True)
        return x

    def forward(self, x, return_attn=False, return_final=False):
        x = self.features(x)

        x = self.protoAT(x, return_attn=return_attn)
        if return_attn:
            return x
        x = self.final_block(x, return_attention=return_final)
        if return_final:
            return x
        x = self.norm(x)
        return x[:,0,:]

def pvit_resnet(num_prototypes=128, prototype_dim=192, patch_size=16, **kwargs):

    model = ProtosimResNet(num_prototypes, prototype_dim, Bottleneck, [3, 4, 6, 3])
    return model

def pvit_tiny(num_prototypes=128, prototype_dim=192, patch_size=16, **kwargs):

    model = ProtosimTransformer(num_prototypes=num_prototypes, prototype_dim=prototype_dim,
            patch_size=patch_size, embed_dim=192, depth=12, num_heads=3, mlp_ratio=4,
        qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model

def pvit_small(num_prototypes=128, prototype_dim=384, patch_size=16, **kwargs):

    model = ProtosimTransformer(num_prototypes=num_prototypes, prototype_dim=prototype_dim,
            patch_size=patch_size, embed_dim=prototype_dim, depth=12, num_heads=6, mlp_ratio=4,
        qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model

class ProtoAttention(nn.Module):
    def __init__(self, num_prototypes, dim, temperature=1, hard=False, gumbel=True):
        super().__init__()
        self.num_prototypes = num_prototypes
        self.temperature = temperature
        self.hard = hard
        self.gumbel = gumbel # for deterministic inference better without gumbel

        self.prototypes = nn.Parameter(torch.rand((num_prototypes, dim)), requires_grad=True)

    def forward(self, x, return_attn=False):

        attn = (self.prototypes @ x.transpose(-2, -1))
        if return_attn:
            return attn

        if self.gumbel:
            attn = F.gumbel_softmax(attn, tau=self.temperature, hard=self.hard, dim=1)
        else:
            dim = 1
            y_soft = attn = attn.softmax(dim=dim)
            if self.hard:
                # Straight through.
                index = y_soft.max(dim, keepdim=True)[1]
                y_hard = torch.zeros_like(attn, memory_format=torch.legacy_contiguous_format).scatter_(dim, index, 1.0)
                attn = y_hard - y_soft.detach() + y_soft
            else:
                # Reparametrization trick.
                attn = y_soft

        x = (attn.transpose(-2,-1) @ self.prototypes)

        return x
