import math
from functools import partial
from einops import rearrange

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.init import constant_

from timm.models.layers import DropPath


class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class DeformableBlock(nn.Module):
    def __init__(self, dim, num_heads, num_samples, qkv_bias=False, drop_path=0., mlp_ratio=2, norm_layer=nn.LayerNorm):
        super().__init__()
        self.num_heads = num_heads
        self.num_samples = num_samples
        head_dim = dim // num_heads
        self.norm1 = norm_layer(dim)
        self.attention_weights = nn.Linear(dim, num_heads * num_samples)
        self.sampling_offsets = nn.Linear(dim, 2 * num_heads * num_samples)
        self.embed_proj = nn.ModuleList([
            nn.Linear(32, head_dim),
            nn.Linear(64, head_dim),
            nn.Linear(128, head_dim),
            nn.Linear(256, head_dim),
            ])

        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=nn.GELU, drop=0.)

        self._reset_parameters()

        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def _reset_parameters(self):
        constant_(self.sampling_offsets.weight.data, 0.)
        thetas = torch.arange(self.num_heads, dtype=torch.float32) * (2.0 * math.pi / self.num_heads)
        grid_init = torch.stack([thetas.cos(), thetas.sin()], -1)
        grid_init = 0.01 * (grid_init / grid_init.abs().max(-1, keepdim=True)[0]).view(self.num_heads, 1, 2).repeat(1, self.num_samples, 1)
        for i in range(self.num_samples):
            grid_init[:, i, :] *= i + 1
        with torch.no_grad():
            self.sampling_offsets.bias = nn.Parameter(grid_init.view(-1))
        constant_(self.attention_weights.weight.data, 0.)
        constant_(self.attention_weights.bias.data, 0.)

    def forward(self, x, ref, features_list):
        """_summary_

        Args:
            x (torch.Tensor(torch.float32)): shape: [(bs, t), 5, 17, 128]
            ref (torch.Tensor(torch.float32)): shape: [(bs, t), 1, 17, 2]
            features_list (torch.Tensor(torch.float32)): shape: [[(bs, t), 256, 8, 6], [(bs, t), 128, 16, 12],
            [(bs, t), 64, 32, 24], [(bs, t), 32, 64, 48]]

        Returns:
            torch.Tensor(torch.float32): _description_
        """
        # [bs, 1, 17, 128], [bs, 4, 17, 128] -> x_0, x
        ## [(bs, t), 1, 17, 128], [(bs, t), 4, 17, 128] -> x_0, x
        x_0, x = x[:, :1], x[:, 1:]
        # bs, 4, 17, _ -> b, l, p, _
        ## (bs t), 4, 17, _ -> b, l, p, _
        b, l, p, _ = x.shape
        # [bs, 4, 17, 128] -> residual
        ## [(bs t), 4, 17, 128] -> residual
        residual = x
        # [bs, 4, 17, 128] + [bs, 1, 17, 1] = [bs, 4, 17, 128]
        ## [(bs t), 4, 17, 128] + [(bs t), 1, 17, 1] = [(bs t), 4, 17, 128]
        x = self.norm1(x + x_0)

        # [bs, 4, 17, 128] -> [bs, 4, 17, 16] -> [bs, 4, 17, 4, 4]
        weights = self.attention_weights(x).view(b, l, p, self.num_heads, self.num_samples)
        # [bs, 4, 17, 4, 4] -> [bs, 4, 17, 4, 4, 1]
        weights = F.softmax(weights, dim=-1).unsqueeze(-1) # b, l, p, num_heads, num_samples, 1
        # [bs, 4, 17, 128] -> [bs, 4, 17, 32] -> [bs, 4, 17, 16, 2]
        offsets = self.sampling_offsets(x).reshape(b, l, p, self.num_heads*self.num_samples, 2).tanh()
        # [bs, 4, 17, 16, 2] + [bs, 1, 17, 1, 2] = [bs, 4, 17, 16, 2]
        pos = offsets + ref.view(b, 1, p, 1, -1)

        # [[bs, 256, 8, 6], [bs, 128, 16, 12], [bs, 64, 32, 24], [bs, 32, 64, 48]] ->
        # [[bs, 256, 17, 16], [bs, 128, 17, 16], [bs, 64, 17, 16], [bs, 32, 17, 16]] ->
        # [[bs, 17, 16, 256], [bs, 17, 16, 128], [bs, 17, 16, 64], [bs, 17, 16, 32]]
        features_sampled = [F.grid_sample(features, pos[:, idx], padding_mode='border', align_corners=True).permute(0, 2, 3, 1).contiguous() \
            for idx, features in enumerate(features_list)]

        # [[bs, 17, 16, 256], [bs, 17, 16, 128], [bs, 17, 16, 64], [bs, 17, 16, 32]] ->
        # [[bs, 17, 16, 32] * 4]
        # b, p, num_heads*num_samples, c
        features_sampled = [embed(features_sampled[idx]) for idx, embed in enumerate(self.embed_proj)]
        # [[bs, 17, 16, 32] * 4] -> [bs, 4, 17, 16, 32]
        features_sampled = torch.stack(features_sampled, dim=1) # b, l, p, num_heads*num_samples, c // num_heads
        # [bs, 4, 17, 4, 4, 1] * ([bs, 4, 17, 16, 32] -> [bs, 4, 17, 4, 4, 32] -> [bs, 4, 17, 4, 4, 1]) = [bs, 4, 17, 4, 4, 1] -> [bs, 4, 17, 16]
        features_sampled = (weights * features_sampled.view(b, l, p, self.num_heads, self.num_samples, -1)).sum(dim=-2).view(b, l, p, -1)
        # [bs, 4, 17, 128] + [bs, 4, 17, 16] = [bs, 4, 17, 128]
        x = residual + self.drop_path(features_sampled)
        # [bs, 4, 17, 128] + [bs, 4, 17, 128] -> [bs, 4, 17, 128] = [bs, 4, 17, 128]
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        # [bs, 5, 17, 128]
        x = torch.cat([x_0,x], dim=1)
        return x