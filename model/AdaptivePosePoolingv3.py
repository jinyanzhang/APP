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
from model.backbones import hrnet
from model.modules.ms_deform_attn import MSDeformAttn
from utils.tools import count_param_numbers


class APPLayer(nn.Module):
    def __init__(self, num_joints=17, n_frames=243, n_heads=8, n_levels=4, n_points=4,
                 dim_img=128, dim_pose=128, drop_path=0.2, drop=.0, mlp_ratio=2., act_layer=nn.GELU):
        super(APPLayer, self).__init__()
        self.n_frames = n_frames
        self.num_joints = num_joints

        assert dim_img % n_heads == 0

        self.self_attn = MSDeformAttn(dim_img, n_levels, n_heads, n_points)
        self.cross_attn = nn.MultiheadAttention(dim_img, n_heads, drop, kdim=dim_pose, vdim=dim_pose, batch_first=True)
        self.mlp = MLP(dim_img, int(dim_img * mlp_ratio), dim_img, act_layer, drop)
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

        self.norm_1 = nn.LayerNorm(dim_img)
        self.norm_2 = nn.LayerNorm(dim_img)
        self.norm_3 = nn.LayerNorm(dim_img)


    def forward(self, x_0, x, f_maps, ref_points, indices, input_shapes):
        b, t, p, c = x.shape
        residual = x
        # x: [bs, 243, 153, 128]
        x = self.norm_1(x)
        # x: [bs, 243, 153, 128]
        x, f_maps = self.self_attn(x, ref_points, f_maps, indices, input_shapes)
        # x: [bs, 243, 153, 128]
        x = self.drop_path(x) + residual

        residual = x
        # x: [bs, 243, 153, 128]
        x = x.view(b*t, p, c)
        # x: [bs, 243, 153, 128]
        x = self.norm_2(x)
        # x: [bs, 243, 153, 128]
        x = self.cross_attn(x, x_0, x_0)[0]
        # x: [bs, 243, 153, 128]
        x = x.view(b, t, p, c)
        x = self.drop_path(x) + residual

        # x: [bs, 243, 153, 128]
        x = self.drop_path(self.mlp(self.norm_3(x))) + x
        return x, f_maps


class AdaptivePosePoolingv3(nn.Module):
    def __init__(self, num_joints=17, n_frames=243, n_layers=4, n_heads=8, n_levels=4, n_points=4, kernel_size=3,
                 drop_path=0.2, dim_img=128, dim_pose=128, dim_out=3, drop=.0, mlp_ratio=4., act_layer=nn.GELU,
                 feature_type='hrnetw32', image_shape=(192, 256)):
        super(AdaptivePosePoolingv3, self).__init__()
        self.k = kernel_size**2

        assert 0 <= drop_path < 1
        drop_path_rates = [x.item() for x in torch.linspace(0, drop_path, n_layers)]

        self.pe_x = nn.Parameter(torch.zeros(1, n_frames, num_joints*self.k, dim_img))
        self.pe_x_0 = nn.Parameter(torch.zeros(1, n_frames, num_joints, dim_pose))
        
        if feature_type == 'hrnetw32':
            backbone_features = [32, 64, 128, 256]
        else:
            backbone_features = [48, 96, 192, 384]
        
        self.input_shapes = torch.tensor([[image_shape[0] // (2**i), image_shape[1] // (2**i)] for i in range(2, 6)], dtype=torch.long)
        self.indices = self.input_shapes.prod(-1)
        self.indices = self.indices.cumsum(0, dtype=torch.long)
        self.indices[1:] = self.indices[:-1].clone()
        self.indices[0] = 0
        self.input_shapes = self.input_shapes.to(torch.float32)
        self.init_offsets = self.__init_offsets(kernel_size)
        
        self.feat_embed = nn.ModuleList([
            nn.Conv2d(backbone_features[0], dim_img, (1, 1)),
            nn.Conv2d(backbone_features[1], dim_img, (1, 1)),
            nn.Conv2d(backbone_features[2], dim_img, (1, 1)),
            nn.Conv2d(backbone_features[3], dim_img, (1, 1)),
        ])

        self.layers = nn.ModuleList([APPLayer(
            num_joints=num_joints, n_frames=n_frames,
            n_heads=n_heads, n_levels=n_levels, n_points=n_points,
            dim_img=dim_img, dim_pose=dim_pose,
            drop_path=drop_path_rates[i], drop=drop, mlp_ratio=mlp_ratio,
            act_layer=act_layer) for i in range(n_layers)])
        
        self.kernel_reduce = nn.Conv1d(dim_img, dim_img, self.k)
        
        self.head = nn.Sequential(
            nn.LayerNorm(dim_img),
            nn.Linear(dim_img, dim_img // 2),
            nn.Tanh(),
            nn.Linear(dim_img // 2, dim_out)
            )

    def __init_offsets(self, kernel_size):
        init_offsets = []
        assert kernel_size // 2
        for j in range(kernel_size):
            for i in range(kernel_size):
                init_offsets += [i - kernel_size // 2, j - kernel_size // 2]
        return torch.Tensor(init_offsets).float()
    
    def forward(self, f_maps, x_0, ref_points):
        """
        Forward Propagation of the decoder

        Args:
            images (torch.Tensor): [bs, t2=9, 3, 384, 288], torch.float32, RGB order, normalized (IMAGENET)
            poses_2d_crop (torch.Tensor): [bs, t1=243, 17, 2], torch.float32, image pixel CS (within 288 * 384)
        """
        b, t1, p, _ = ref_points.shape
        t2 = f_maps[0].shape[1]
        t3 = t1 // t2
        assert t2 * t3 == t1
        device = x_0.device
        
        indices = self.indices.to(device)
        input_shapes = self.input_shapes.to(device)
        init_offsets = self.init_offsets.to(device)
        
        f_maps = [rearrange(f, 'b t c h w -> (b t) c h w') for f in f_maps]

        ref_points = ref_points.repeat(1, 1, 1, self.k)
        ref_points += init_offsets[None, None, None, :]
        ref_points = ref_points.view(b*t2, t3, -1, 2)
        pk = ref_points.shape[-2]
        # normalize, map to [-1, 1]
        init_ref_points = ref_points.clone()
        init_ref_points[..., :2] /= torch.tensor([288//2, 384//2], device=device)
        init_ref_points[..., :2] -= torch.tensor([1, 1], device=device)
        # normalize, map to [0, 1]
        ref_points = rearrange(ref_points, 'bt2 t3 pk c -> bt2 (t3 pk) c')
        ref_points[..., :2] /= torch.tensor([288, 384], device=device)
        ref_points = ref_points[:, :, None, :]

        f_maps = [self.feat_embed[i](f) for i, f in enumerate(f_maps)]
        
        x = [F.grid_sample(f, init_ref_points, mode='bilinear', padding_mode='zeros', align_corners=False).clone() for f in f_maps]
        x = [rearrange(f, '(b t2) c t3 pk -> b (t2 t3) pk c', t2=t2) for f in x]
        x = torch.stack(x, dim=-2).mean(-2)

        f_maps = torch.cat([rearrange(f, 'bt2 c h w -> bt2 (h w) c') for f in f_maps], dim=1)

        x = self.pe_x + x
        x_0 = self.pe_x_0 + x_0
        x_0 = x_0.view(b*t1, p, -1)
        for layer in self.layers:
            x, f_maps = layer(x_0, x, f_maps, ref_points, indices, input_shapes)
        x = rearrange(x, 'b t (p k) c -> (b t p) c k', p=p, k=self.k)
        x = self.kernel_reduce(x)
        x = x.view(b, t1, p, -1)
        x = self.head(x)
        return x

def _test():
    
    from torchprofile import profile_macs
    import warnings
    warnings.filterwarnings('ignore')
    b, c, t, j = 1, 3, 243, 17
    poses = torch.randn((b, t, j, 128)).cuda()
    f_maps = [
        torch.randn((1, 27, 32, 64, 48)).cuda(),
        torch.randn((1, 27, 64, 32, 24)).cuda(),
        torch.randn((1, 27, 128, 16, 12)).cuda(),
        torch.randn((1, 27, 256, 8, 6)).cuda(),
        ]
    poses_crop = torch.randint(low=0, high=256, size=(b, t, j, c - 1)).float().cuda()

    model = AdaptivePosePoolingv3().cuda()
    model.eval()

    n_params = count_param_numbers(model) / 1000000
    print(f"Model parameter #: {n_params:.2f}M")
    print(f"Model MACs #: {profile_macs(model, (f_maps, poses, poses_crop)):,}")

    # Warm-up to avoid timing fluctuations
    for _ in range(10):
        _ = model(f_maps, poses, poses_crop)

    import time
    num_iterations = 100
    # Measure the inference time for 'num_iterations' iterations
    start_time = time.time()
    for _ in range(num_iterations):
        with torch.no_grad():
            _ = model(f_maps, poses, poses_crop)
    end_time = time.time()

    # Calculate the average inference time per iteration
    average_inference_time = (end_time - start_time) / num_iterations

    # Calculate FPS
    fps = 1.0 / average_inference_time
    print(f"Time: {average_inference_time}")
    print(f"FPS: {fps}")

    out = model(f_maps, poses, poses_crop)

    assert out.shape == (b, t, j, 3), f"Output shape should be {b}x{t}x{j}x3 but it is {out.shape}"


if __name__ == '__main__':
    _test()
