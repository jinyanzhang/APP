# ------------------------------------------------------------------------------------------------
# Deformable DETR
# Copyright (c) 2020 SenseTime. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------------------------------
# Modified from https://github.com/chengdazhi/Deformable-Convolution-V2-PyTorch/tree/pytorch_1.0.0
# ------------------------------------------------------------------------------------------------

from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import warnings
import math

import torch
import torch.nn.functional as F

from torch import nn
from torch.nn.init import xavier_uniform_, constant_
from einops import rearrange

from model.modules.ms_deform_attn_func import MSDeformAttnFunction


def _is_power_of_2(n):
    if (not isinstance(n, int)) or (n < 0):
        raise ValueError("invalid input for _is_power_of_2: {} (type: {})".format(n, type(n)))
    return (n & (n-1) == 0) and n != 0


class MSDeformAttn(nn.Module):
    def __init__(self, d_model=128, n_levels=4, n_heads=8, n_points=4):
        """
        Multi-Scale Deformable Attention Module
        :param d_model      hidden dimension
        :param n_levels     number of feature levels
        :param n_heads      number of attention heads
        :param n_points     number of sampling points per attention head per feature level
        """
        super().__init__()
        if d_model % n_heads != 0:
            raise ValueError('d_model must be divisible by n_heads, but got {} and {}'.format(d_model, n_heads))
        _d_per_head = d_model // n_heads
        # you'd better set _d_per_head to a power of 2 which is more efficient in our CUDA implementation
        if not _is_power_of_2(_d_per_head):
            warnings.warn("You'd better set d_model in MSDeformAttn to make the dimension of each attention head a power of 2 "
                          "which is more efficient in our CUDA implementation.")

        self.im2col_step = 27 # initially is 64

        self.d_model = d_model
        self.n_levels = n_levels
        self.n_heads = n_heads
        self.n_points = n_points
        self.shapes = torch.ones((4, 2), dtype=torch.long)

        self.sampling_offsets = nn.Linear(d_model, n_heads * n_levels * n_points * 2)
        self.attention_weights = nn.Linear(d_model, n_heads * n_levels * n_points)
        self.value_proj = nn.Linear(d_model, d_model)
        self.output_proj = nn.Linear(d_model, d_model)

        self._reset_parameters()

    def _reset_parameters(self):
        constant_(self.sampling_offsets.weight.data, 0.)
        thetas = torch.arange(self.n_heads, dtype=torch.float32) * (2.0 * math.pi / self.n_heads)
        grid_init = torch.stack([thetas.cos(), thetas.sin()], -1)
        grid_init = (grid_init / grid_init.abs().max(-1, keepdim=True)[0]).view(self.n_heads, 1, 1, 2).repeat(1, self.n_levels, self.n_points, 1)
        for i in range(self.n_points):
            grid_init[:, :, i, :] *= i + 1
        with torch.no_grad():
            self.sampling_offsets.bias = nn.Parameter(grid_init.view(-1))
        constant_(self.attention_weights.weight.data, 0.)
        constant_(self.attention_weights.bias.data, 0.)
        xavier_uniform_(self.value_proj.weight.data)
        constant_(self.value_proj.bias.data, 0.)
        xavier_uniform_(self.output_proj.weight.data)
        constant_(self.output_proj.bias.data, 0.)

    def forward(self, query, ref_points, value, indices, input_shapes):
        """
        :param query                       [bs, 243, 17, 128]
        :param ref_points                  [(bs, 27), 9*17=153, 4(1), 2], range in [0, 1], top-left (0,0), bottom-right (1, 1), including padding area
        :param value                       [(bs, 27), 64*48+32*24+16*12+8*6=4080, 128], (bs, \sum_{l=0}^{L-1} H_l \cdot W_l, C)
        :param indices                     [0, 3072, 3072+768=3840, 3072+768+192=4032]
                                           [0, H_0*W_0, H_0*W_0+H_1*W_1, H_0*W_0+H_1*W_1+H_2*W_2, ..., H_0*W_0+H_1*W_1+...+H_{L-1}*W_{L-1}]
        :param input_shapes                [[48, 64], [24, 32], [12, 16], [6, 8]]
                                           [[W_0, H_0], [W_1, H_1], [W_2, H_2], [W_3, H_3]]

        :return output                     [bs, 243, 17, 128]
        """
        b, t1, p, c = query.shape
        bt2, l_v, _ = value.shape
        t2 = bt2 // b
        t3 = t1 // t2
        # query: [bs, 243, 17, 128] -> [(bs, 27), 153, 128]
        query = rearrange(query, 'b (t2 t3) p c -> (b t2) (t3 p) c', t2=t2, t3=t3)
        
        device = value.device
        
        assert input_shapes.prod(-1).sum() == value.shape[1]

        # new_value: [(bs, 27), 4080, 128] -> [(bs, 27), 4080, 128]
        value = self.value_proj(value)
        new_value = value
        # value: [(bs, 27), 4080, 128] -> [(bs, 27), 4080, 8, 128//8=16]
        value = value.view(bt2, l_v, self.n_heads, self.d_model // self.n_heads)
        # query: [(bs, 27), 153, 128] -> sampling_offsets: [(bs, 27), 153, 8*4*4*2=256] -> [(bs, 27), 153, 8, 4, 4, 2]
        sampling_offsets = self.sampling_offsets(query).view(bt2, t3*p, self.n_heads, self.n_levels, self.n_points, 2)
        # query: [(bs, 27), 153, 128] -> attention_weights: [(bs, 27), 153, 8*4*4=128] -> [(bs, 27), 153, 8, 4*4=16]
        attention_weights = self.attention_weights(query).view(bt2, t3*p, self.n_heads, self.n_levels * self.n_points)
        # attention_weights: [(bs, 27), 153, 8, 16] -> [(bs, 27), 153, 8, 16] -> [(bs, 27), 153, 8, 4, 4]
        attention_weights = F.softmax(attention_weights, -1).view(bt2, t3*p, self.n_heads, self.n_levels, self.n_points)
        # offset_normalizer: [4, 2] -> [1, 1, 1, 4, 1, 2]
        offset_normalizer = input_shapes[None, None, None, :, None, :]
        # sampling_offsets / offset_normalizer: [(bs, 27), 153, 8, 4, 4, 2]
        sampling_offsets = sampling_offsets / offset_normalizer
        # sampling_locations: [(bs, 27), 153, 8, 4, 4, 2] + [(bs, 27), 153, 1, 1, 1, 2] -> [(bs, 27), 4080, 8, 4, 4, 2]
        sampling_locations = ref_points[:, :, None, :, None, :] + sampling_offsets
        # new_query: [(bs, 27), 153, 128]
        new_query = MSDeformAttnFunction.apply(value, self.shapes.to(device), indices, sampling_locations, attention_weights, self.im2col_step)
        # new_query: [(bs, 27), 153, 128] -> [(bs, 27), 153, 128]
        new_query = self.output_proj(new_query)
        # new_query: [(bs, 27), 153, 128] -> [bs, 243, 17, 128]
        new_query = rearrange(new_query, '(b t2) (t3 p) c -> b (t2 t3) p c', b=b, t2=t2, t3=t3, p=p)
        return new_query, new_value
    
    
if __name__ == '__main__':
    def_attn = MSDeformAttn(128, 4, 8, 4).cuda()
    query = torch.randn((27, 153, 128), dtype=torch.float32, device='cuda')
    value = torch.randn((27, 4080, 128), dtype=torch.float32, device='cuda')
    ref_points = torch.randn((27, 153, 1, 2), dtype=torch.float32, device='cuda')
    indices = torch.tensor([0, 3072, 3840, 4032], dtype=torch.long, device='cuda')
    out = def_attn(query, ref_points, value, indices)
    print(out.shape)