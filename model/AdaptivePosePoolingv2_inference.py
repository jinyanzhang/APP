import math
import torch
from torch import nn
import torch.nn.functional as F

from einops import rearrange, reduce
from timm.models.layers import DropPath

import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from model.modules.mlp import MLP


class DeformableConv(nn.Module):
    def __init__(self, kernel_size=3, alpha=0.9, dim_img=128,
                 backbone_features=[32, 64, 128, 256]):
        super(DeformableConv, self).__init__()
        assert 1 > alpha > 0
        self.alpha = alpha
        assert kernel_size > 0 and isinstance(kernel_size, int)
        self.kernel_size = kernel_size

        self.sampling_offsets = nn.Linear(dim_img, 2)
        self.attn_weights = nn.Linear(dim_img, 2)
        
        self.feat_embed = nn.ModuleList([
            nn.Conv2d(backbone_features[0], dim_img, (1, 1)),
            nn.Conv2d(backbone_features[1], dim_img, (1, 1)),
            nn.Conv2d(backbone_features[2], dim_img, (1, 1)),
            nn.Conv2d(backbone_features[3], dim_img, (1, 1)),
        ])
        
        self.conv = nn.Conv1d(dim_img, dim_img, kernel_size**2)

    def forward(self, f_maps, query, key, value):
        """_summary_

        Args:
            f_maps (torch.Tensor):[(bs, 9), 48/96/192/384, 96/48/24/12, 72/36/18/9]
            query (torch.Tensor): [bs, 243, 17*9=153, 128]
            key (torch.Tensor):   [(bs, 9), 27, 153, 2]
            value (torch.Tensor): [bs, 243, 17, 128]

        Returns:
            _type_: _description_
        """
        b, t1, p, _ = value.shape
        t3 = key.shape[1]
        t2 = t1 // t3
        # sampling_offsets: [bs, 243, 153, 128] -> [bs, 243, 153, 2]
        sampling_offsets = self.sampling_offsets(query).tanh()
        
        # attn_weights: [bs, 243, 17, 128]*[bs, 243, 17, 1] -> [bs, 243, 17, 2]
        attn_weights = F.softmax(self.attn_weights(value), dim=-1)
        # attn_weights: [bs, 243, 17, 2] -> [bs, 243, 153, 2]
        attn_weights = attn_weights.repeat(1, 1, self.kernel_size*self.kernel_size, 1)
        # sampling_positions: [bs, 243, 153, 2]+[bs, 243, 153, 2]*[bs, 243, 153, 2] -> [bs, 243, 153, 2]
        sampling_positions = sampling_offsets * attn_weights
        
        sampling_offsets = sampling_positions.view(b*t2, t3, -1, 2)
        sampling_positions = key + sampling_offsets
        # new_query: [[(bs, 9), 48/96/192/384, 96/48/24/12, 72/36/18/9]] ->
        # [[(bs, 9), 128, 96/48/24/12, 72/36/18/9]] ->
        # [[(bs, 9), 128, 27, 153]] ->
        # [[bs, 243, 153, 128]]
        new_query = [F.grid_sample(f, sampling_positions, align_corners=False, padding_mode='zeros') for f in f_maps]
        new_query = [self.feat_embed[i](f) for i, f in enumerate(new_query)]
        new_query = [rearrange(f, '(b t2) c t3 p -> b (t2 t3) p c', t2=t2) for f in new_query]
        # new_query: [bs, 243, 153, 128]
        new_query = torch.stack(new_query, dim=-2)
        new_query = new_query.mean(dim=-2)
        # new_query: [bs, 243, 153, 128]
        new_query = (1 - self.alpha) * new_query + self.alpha * query
        # new_value: [bs, 243, 153, 128] -> [(bs, 243, 17), 128, 9]
        new_value = rearrange(new_query, 'b t (p k) c -> (b t p) c k', k=self.kernel_size**2)
        # new_value: [(bs, 243, 17), 128, 9] -> [(bs, 243, 17), 128, 1]
        new_value = self.conv(new_value)
        # new_value: [(bs, 243, 17), 128, 1, 1] -> [bs, 243, 17, 128]
        new_value = rearrange(new_value, '(b t p) c 1 -> b t p c', b=b, t=t1, p=p)
        return new_query, new_value, sampling_positions


class APPLayer(nn.Module):
    def __init__(self, num_joints=17, n_frames=243, n_heads=8, kernel_size=3, alpha=0.9,
                 dim_img=128, dim_pose=128, drop_path=0.2, drop=.0, mlp_ratio=2., act_layer=nn.GELU,
                 backbone_features=[32, 64, 128, 256]):
        super(APPLayer, self).__init__()
        self.n_frames = n_frames
        self.num_joints = num_joints

        assert dim_img % n_heads == 0

        self.def_conv = DeformableConv(kernel_size, alpha, dim_img, backbone_features)
        self.cross_attn = nn.MultiheadAttention(dim_img, n_heads, drop, kdim=dim_pose, vdim=dim_pose, batch_first=True)
        self.mlp = MLP(dim_img, int(dim_img * mlp_ratio), dim_img, act_layer, drop)
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

        self.norm_1 = nn.LayerNorm(dim_img)
        self.norm_2 = nn.LayerNorm(dim_img)
        self.norm_3 = nn.LayerNorm(dim_img)


    def forward(self, x_0, f_maps, new_f_maps, ref_points, x):
        b, t, p, c = x.shape
        residual = x
        # x: [bs, 243, 17, 128]
        x = self.norm_1(x)
        # x: [bs, 243, 17, 128]
        new_f_maps, x, points = self.def_conv(f_maps, new_f_maps, ref_points, x)
        # x: [bs, 243, 17, 128]
        x = self.drop_path(x) + residual

        residual = x
        # x: [bs, 243, 17, 128]
        x = x.view(b*t, p, c)
        # x: [bs, 243, 17, 128]
        x = self.norm_2(x)
        # x: [bs, 243, 17, 128]
        x, attn_map = self.cross_attn(x, x_0, x_0)
        # x: [bs, 243, 17, 128]
        x = x.view(b, t, p, c)
        x = self.drop_path(x) + residual

        # x: [bs, 243, 17, 128]
        x = self.drop_path(self.mlp(self.norm_3(x))) + x
        return x, new_f_maps, attn_map, points


class AdaptivePosePoolingv2(nn.Module):
    def __init__(self, num_joints=17, n_frames=243, n_layers=4, n_heads=8, kernel_size=3, alpha=0.9, drop_path=0.2,
                 dim_img=128, dim_pose=128, dim_out=3, drop=.0, mlp_ratio=4., act_layer=nn.GELU,
                 backbone_features=[32, 64, 128, 256]):
        super(AdaptivePosePoolingv2, self).__init__()
        self.k = kernel_size**2

        assert 0 <= drop_path < 1
        drop_path_rates = [x.item() for x in torch.linspace(0, drop_path, n_layers)]

        self.pe_x = nn.Parameter(torch.zeros(1, n_frames, num_joints, dim_img))
        self.pe_x_0 = nn.Parameter(torch.zeros(1, n_frames, num_joints, dim_pose))
        
        
        
        self.feat_embed = nn.ModuleList([
            nn.Conv2d(backbone_features[0], dim_img, (1, 1)),
            nn.Conv2d(backbone_features[1], dim_img, (1, 1)),
            nn.Conv2d(backbone_features[2], dim_img, (1, 1)),
            nn.Conv2d(backbone_features[3], dim_img, (1, 1)),
        ])

        self.decoders = nn.ModuleList([APPLayer(
            num_joints=num_joints, n_frames=n_frames,
            n_heads=n_heads, kernel_size=kernel_size,
            alpha=alpha, dim_img=dim_img, dim_pose=dim_pose,
            drop_path=drop_path_rates[i], drop=drop,
            mlp_ratio=mlp_ratio, act_layer=act_layer,
            backbone_features=backbone_features) for i in range(n_layers)])
        self.init_offsets = self.__init_offsets(kernel_size)
        
        self.head = nn.Sequential(
            nn.LayerNorm(dim_img),
            nn.Linear(dim_img, dim_img // 2),
            nn.Tanh(),
            nn.Linear(dim_img // 2, dim_out)
            )

    def __init_offsets(self, kernel_size):
        init_offsets = []
        for j in range(kernel_size):
            for i in range(kernel_size):
                init_offsets += [i - kernel_size // 2, j - kernel_size // 2]
        return torch.Tensor(init_offsets).float()

    def forward(self, f_maps, x_0, ref_points):
        b, t1, p, _ = ref_points.shape
        t2 = f_maps[0].shape[1]
        t3 = t1 // t2
        assert t2 * t3 == t1
        device = x_0.device
        
        f_maps = [rearrange(f, 'b t c h w -> (b t) c h w') for f in f_maps]

        # normalize, map to [0, 1], for multiscale deformable attention
        init_offsets = self.init_offsets.to(device)
        # ref_points: [bs, 243, 17, 2] -> [bs, 243, 17, 18]
        ref_points = ref_points.repeat(1, 1, 1, init_offsets.shape[0] // 2)
        # ref_points: [bs, 243, 17, 2] -> [bs, 243, 17, 18]
        ref_points += init_offsets[None, None, None, :]
        # ref_points: [bs, 243, 17, 18] -> [(bs, 9), 27, 17*9=153, 2]
        ref_points = ref_points.view(b*t2, t3, -1, 2)
        
        points = []
        points.append(ref_points)
        
        # normalize, map to [-1, 1]
        ref_points[..., :2] /= torch.tensor([48//2, 64//2], device=device)
        ref_points[..., :2] -= torch.tensor([1, 1], device=device)

        new_f_maps = [F.grid_sample(f, ref_points, align_corners=False, padding_mode='zeros').clone() for f in f_maps]
        new_f_maps = [self.feat_embed[i](f) for i, f in enumerate(new_f_maps)]
        new_f_maps = [rearrange(f, '(b t2) c t3 p -> b (t2 t3) p c', t2=t2) for f in new_f_maps]

        # new_f_maps: [bs, 243, 153, 4, 128]
        new_f_maps = torch.stack(new_f_maps, dim=-2)
        # new_f_maps: [bs, 243, 153, 128]
        new_f_maps = new_f_maps.mean(-2)
        # x: [bs, 243, 153, 4, 128]
        x = reduce(new_f_maps, 'b t (p k) c -> b t p c', 'mean', k=self.k)

        x = self.pe_x + x
        x_0 = self.pe_x_0 + x_0
        x_0 = x_0.view(b*t1, p, -1)
        
        attn_maps = []
        for decoder in self.decoders:
            x, new_f_maps, attn_map, sampling_points = decoder.forward(x_0, f_maps, new_f_maps, ref_points, x)
            attn_maps.append(attn_map)
            points.append(sampling_points)
        attn_maps = torch.stack(attn_maps, dim=1)
        points = torch.stack(points, dim=1)
        x = self.head(x)
        return x, attn_maps, points
