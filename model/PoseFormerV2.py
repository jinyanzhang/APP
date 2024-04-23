import math
import logging
from functools import partial
from einops import rearrange

import torch
import torch_dct as dct
import torch.nn as nn
import torch.nn.functional as F

import numpy as np
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


class FreqMlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        b, f, _ = x.shape
        x = dct.dct(x.permute(0, 2, 1)).permute(0, 2, 1).contiguous()
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        x = dct.idct(x.permute(0, 2, 1)).permute(0, 2, 1).contiguous()
        return x


class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        # NOTE scale factor was wrong in my original version, can set manually to be compat with prev weights
        self.scale = qk_scale or head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]   # make torchscript happy (cannot use tensor as tuple)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class Block(nn.Module):

    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(
            dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def forward(self, x):
        x = x + self.drop_path(self.attn(self.norm1(x)))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


class MixedBlock(nn.Module):

    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(
            dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp1 = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)
        self.norm3 = norm_layer(dim)
        self.mlp2 = FreqMlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def forward(self, x):
        b, f, c = x.shape
        x = x + self.drop_path(self.attn(self.norm1(x)))
        x1 = x[:, :f//2] + self.drop_path(self.mlp1(self.norm2(x[:, :f//2])))
        x2 = x[:, f//2:] + self.drop_path(self.mlp2(self.norm3(x[:, f//2:])))
        return torch.cat((x1, x2), dim=1)


class PoseTransformerV2(nn.Module):
    def __init__(self, in_chans=2, depth=4, embed_dim_ratio=32,
                 number_of_kept_frames=27, number_of_kept_coeffs=27,
                 num_heads=8, mlp_ratio=2., qkv_bias=True, qk_scale=None,
                 drop_rate=0., attn_drop_rate=0., drop_path_rate=0.2,
                 norm_layer=None, n_frames=243, num_joints=17,
                 return_mid=False):
        """    ##########hybrid_backbone=None, representation_size=None,
        Args:
            n_frames (int, tuple): input frame number
            num_joints (int, tuple): joints number
            in_chans (int): number of input channels, 2D joints have 2 channels: (x,y)
            embed_dim_ratio (int): embedding dimension ratio
            depth (int): depth of transformer
            num_heads (int): number of attention heads
            mlp_ratio (int): ratio of mlp hidden dim to embedding dim
            qkv_bias (bool): enable bias for qkv if True
            qk_scale (float): override default qk scale of head_dim ** -0.5 if set
            drop_rate (float): dropout rate
            attn_drop_rate (float): attention dropout rate
            drop_path_rate (float): stochastic depth rate
            norm_layer: (nn.Module): normalization layer
        """
        super().__init__()

        norm_layer = norm_layer or partial(nn.LayerNorm, eps=1e-6)
        embed_dim = embed_dim_ratio * num_joints   #### temporal embed_dim is num_joints * spatial embedding dim ratio
        out_dim = num_joints * 3    #### output dimension is num_joints * 3
        
        self.in_chans = in_chans
        
        self.num_frame_kept = number_of_kept_frames
        self.num_coeff_kept = number_of_kept_coeffs if number_of_kept_coeffs else self.num_frame_kept

        self.return_mid = return_mid

        ### spatial patch embedding
        self.Joint_embedding = nn.Linear(in_chans, embed_dim_ratio)
        self.Freq_embedding = nn.Linear(in_chans*num_joints, embed_dim)

        self.Spatial_pos_embed = nn.Parameter(torch.zeros(1, num_joints, embed_dim_ratio))
        self.Temporal_pos_embed = nn.Parameter(torch.zeros(1, self.num_frame_kept, embed_dim))
        self.Temporal_pos_embed_ = nn.Parameter(torch.zeros(1, self.num_coeff_kept, embed_dim))
        self.pos_drop = nn.Dropout(p=drop_rate)

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]  # stochastic depth decay rule

        self.Spatial_blocks = nn.ModuleList([
            Block(
                dim=embed_dim_ratio, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer)
            for i in range(depth)])

        self.blocks = nn.ModuleList([
            MixedBlock(
                dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer)
            for i in range(depth)])

        self.Spatial_norm = norm_layer(embed_dim_ratio)
        self.Temporal_norm = norm_layer(embed_dim)

        ####### A easy way to implement weighted mean
        self.weighted_mean = torch.nn.Conv1d(in_channels=self.num_coeff_kept, out_channels=1, kernel_size=1)
        self.weighted_mean_ = torch.nn.Conv1d(in_channels=self.num_frame_kept, out_channels=1, kernel_size=1)

        self.head = nn.Sequential(
            nn.LayerNorm(embed_dim*2),
            nn.Linear(embed_dim*2, out_dim),
        )

    def Spatial_forward_features(self, x):
        b, f, p, _ = x.shape  ##### b is batch size, f is number of frames, p is number of joints
        num_frame_kept = self.num_frame_kept

        index = torch.arange((f-1)//2-num_frame_kept//2, (f-1)//2+num_frame_kept//2+1)

        x = self.Joint_embedding(x[:, index].view(b*num_frame_kept, p, -1))
        x += self.Spatial_pos_embed
        x = self.pos_drop(x)

        for blk in self.Spatial_blocks:
            x = blk(x)

        x = self.Spatial_norm(x)
        x = rearrange(x, '(b f) p c -> b f (p c)', f=num_frame_kept)
        return x

    def forward_features(self, x, Spatial_feature):
        b, f, p, _ = x.shape
        num_coeff_kept = self.num_coeff_kept

        x = dct.dct(x.permute(0, 2, 3, 1))[:, :, :, :num_coeff_kept]
        x = x.permute(0, 3, 1, 2).contiguous().view(b, num_coeff_kept, -1)
        x = self.Freq_embedding(x) 
        
        Spatial_feature += self.Temporal_pos_embed
        x += self.Temporal_pos_embed_
        x = torch.cat((x, Spatial_feature), dim=1)

        for blk in self.blocks:
            x = blk(x)

        x = self.Temporal_norm(x)
        return x

    def forward(self, x):
        x = x[..., :self.in_chans]
        b, f, p, _ = x.shape
        x_ = x.clone()

        Spatial_feature = self.Spatial_forward_features(x)
        x = self.forward_features(x_, Spatial_feature)
        
        # [bs, 54, 544=32*17]
        if self.return_mid:
            return x
        
        # [bs, 54, 544=32*17] ->  [bs, 27, 544=32*17]
        x = torch.cat((self.weighted_mean(x[:, :self.num_coeff_kept]), self.weighted_mean_(x[:, self.num_coeff_kept:])), dim=-1)
        
        x = self.head(x).view(b, 1, p, -1)
        return x


def _test():
    from torchprofile import profile_macs
    import warnings
    warnings.filterwarnings('ignore')
    b, c, t, j = 1, 3, 243, 17
    random_x = torch.randn((b, t, j, c)).cuda()

    model = PoseTransformerV2().cuda()
    
    # checkpoint = torch.load('/data-home/checkpoints/27_243_45.2.bin')
    # state_dict = {}
    # for key in checkpoint['model_pos'].keys():
    #     state_dict[key.replace('module.', '')] = checkpoint['model_pos'][key].clone()
    # new_checkpoint = {'model': state_dict}
    # torch.save(new_checkpoint, 'poseformerv2-h36m-cpn-author.pth.tr')
    
    state_dict = torch.load('/data-home/checkpoints/poseformerv2-h36m-cpn-author.pth.tr')['model']
    ret = model.load_state_dict(state_dict, strict=True)
    print(ret)
    model.eval()

    model_params = 0
    for parameter in model.parameters():
        model_params = model_params + parameter.numel()
    print(f"Model parameter #: {model_params / 1000000:.2f}M")
    print(f"Model MACs #: {profile_macs(model, random_x):,}")

    # Warm-up to avoid timing fluctuations
    for _ in range(10):
        _ = model(random_x)

    import time
    num_iterations = 100 
    # Measure the inference time for 'num_iterations' iterations
    start_time = time.time()
    for _ in range(num_iterations):
        with torch.no_grad():
            _ = model(random_x)
    end_time = time.time()

    # Calculate the average inference time per iteration
    average_inference_time = (end_time - start_time) / num_iterations

    # Calculate FPS
    fps = 1.0 / average_inference_time

    print(f"FPS: {fps}")
    

    out = model(random_x)

    assert out.shape == (b, 1, j, 3), f"Output shape should be {b}x1x{j}x3 but it is {out.shape}"


if __name__ == '__main__':
    _test()