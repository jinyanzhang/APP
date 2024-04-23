import math
import torch
from torch import nn
import torch.nn.functional as F

from einops import rearrange, repeat
from timm.models.layers import DropPath

import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from model.modules.mlp import MLP
from model.modules.ms_deform_attn import MSDeformAttn
from model.backbones import hrnet
from model.MotionAGFormer import MotionAGFormer as MotionAGTransformerEncoder
from utils.tools import count_param_numbers

class PositionalEncoding(nn.Module):
    def __init__(self, num_joints=17, n_frames=243, n_levels=4, dim_img=128, dropout=0):
        super().__init__()
        self.num_joints = num_joints
        self.n_frames = n_frames
        self.n_levels = n_levels

        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(n_frames).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, dim_img, 2) * (-math.log(10000.0) / dim_img))
        # [1, 4, 243, 17, 128]
        pe = torch.zeros(1, n_frames, num_joints, n_levels, dim_img)
        pe[..., 0::2] = torch.sin((position * div_term)[None, :, None, None, :])
        pe[..., 1::2] = torch.cos((position * div_term)[None, :, None, None, :])
        self.register_buffer('pe', pe)

        self.learn_pos = nn.Parameter(torch.zeros(1, n_frames, num_joints, n_levels, dim_img))

    def forward(self, x):
        x = rearrange(x, 'b (t p l) c -> b t p l c', l=self.n_levels, t=self.n_frames, p=self.num_joints)
        x = x + self.pe[:, :x.size(1)] + self.learn_pos
        x = rearrange(x, 'b t p l c -> b (t p l) c')
        return self.dropout(x)


class MotionAGTransformerDecoderLayer(nn.Module):
    def __init__(self, num_joints=17, n_frames=243, n_levels=4, n_heads=8, n_points=4, drop_path=0.2,
                 dim_img=128, drop=.0, mlp_ratio=4., act_layer=nn.GELU):
        super(MotionAGTransformerDecoderLayer, self).__init__()
        self.n_frames = n_frames
        self.n_levels = n_levels
        self.num_joints = num_joints

        self.pe = PositionalEncoding(num_joints, n_frames, n_levels, dim_img, drop)
        self.self_attn = MSDeformAttn(dim_img, n_levels, n_heads, n_points)
        self.cross_attn = nn.MultiheadAttention(dim_img, n_heads, drop, batch_first=True)
        self.mlp = MLP(dim_img, int(dim_img * mlp_ratio), dim_img, act_layer, drop)
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

        self.norm_1 = nn.LayerNorm(dim_img)
        self.norm_2 = nn.LayerNorm(dim_img)
        self.norm_3 = nn.LayerNorm(dim_img)

        self.indices = torch.LongTensor([i * num_joints * n_frames for i in range(n_levels)])

    def subforward_self(self, x, ref_points):
        device = x.device
        indices = self.indices.to(device)
        # x: [bs, 16524, 128]
        x = self.norm_1(x)
        # x: [bs, 16524, 128] -> [bs, 16524, 128]
        x = self.self_attn(self.pe(x), ref_points, x, indices)
        return x

    def subforward_cross(self, x):
        x = self.norm_2(x)
        x = self.cross_attn(x, x, x)[0]
        return x

    def forward(self, x, ref_points):
        """_summary_

        Args:
            x (torch.Tensor): [bs, 243, 17, 5, 128]
            ref_points (torch.Tensor): [bs, 243, 17, 2]

        Raises:
            x (torch.Tensor): [bs, 243, 17, 5, 128]
        """
        b, t, p, l, c = x.shape
        x_0, x = x[..., :1, :], x[..., 1:, :]
        x = rearrange(x, 'b t p l c -> b (t p l) c')
        # x: [bs, 16524, 128]
        x = self.drop_path(self.subforward_self(x, ref_points)) + x

        x = x.view(b, t, p, l-1, c)
        x = torch.cat([x_0, x], dim=-2).view(b*t, p*l, c)
        x = self.drop_path(self.subforward_cross(x)) + x

        x = x.view(b, t, p, l, c)
        # x: [bs, 16524, 128]
        x = self.drop_path(self.mlp(self.norm_3(x))) + x
        return x


class MotionAGTransformerDecoder(nn.Module):
    def __init__(self, num_joints=17, n_frames=243, n_layers=6, n_levels=4, n_heads=8, n_points=4, drop_path=0.2,
                 dim_img=128, drop=.0, mlp_ratio=4., act_layer=nn.GELU,
                 backbone_feats=[32, 64, 128, 256],
                 backbone_type='hrnet', backbone_name='hrnet_w32_384_288',
                 backbone_pretrained='/home/xxxxxx/MotionAGFormer/data/pretrained/coco/pose_hrnet_w32_384x288.pth'):
        super(MotionAGTransformerDecoder, self).__init__()
        self.dim_img = dim_img
        self.n_levels = n_levels

        self.backbone = eval(f'{backbone_type}.build_model(backbone_name, False)')
        self.backbone = eval(f'{backbone_type}.load_model(self.backbone, backbone_pretrained)')

        # set requires_grad = False
        for p in self.backbone.parameters():
            p.requires_grad = False

        assert len(backbone_feats) == 4
        self.feat_emb = nn.ModuleList([
            nn.Conv2d(backbone_feats[0], dim_img, kernel_size=(1, 1)),
            nn.Conv2d(backbone_feats[1], dim_img, kernel_size=(1, 1)),
            nn.Conv2d(backbone_feats[2], dim_img, kernel_size=(1, 1)),
            nn.Conv2d(backbone_feats[3], dim_img, kernel_size=(1, 1)),
            ])

        assert 0 <= drop_path < 1
        drop_path_rates = [x.item() for x in torch.linspace(0, drop_path, n_layers)]

        self.decoders = nn.ModuleList([MotionAGTransformerDecoderLayer(
            num_joints=num_joints, n_frames=n_frames, n_levels=n_levels,
            n_heads=n_heads, n_points=n_points, drop_path=drop_path_rates[i],
            dim_img=dim_img, drop=drop, mlp_ratio=mlp_ratio, act_layer=act_layer) for i in range(n_layers)])

    def forward(self, images, x_0, ref_poses):
        """
        Forward Propagation of the decoder

        Args:
            images (torch.Tensor): [bs, t2=9, 3, 384, 288], torch.float32, RGB order, normalized (IMAGENET)
            poses_2d_crop (torch.Tensor): [bs, t1=243, 17, 2], torch.float32, image pixel CS (within 288 * 384)
        """
        assert images.ndim == 5
        assert ref_poses.ndim == 4
        b, t2, *_ = images.shape
        _, t1, p, _ = ref_poses.shape
        t3 = t1 // t2
        assert t2 * t3 == t1
        device = images.device

        images = rearrange(images, 'b t c h w -> (b t) c h w').contiguous()
        # normalize, map to [0, 1], for multiscale deformable attention
        ref_points = ref_poses.clone() / torch.tensor([288, 384], device=device)
        # normalize, map to [-1, 1]
        ref_poses[..., :2] /= torch.tensor([288//2, 384//2], device=device)
        ref_poses[..., :2] -= torch.tensor([1, 1], device=device)
        ref_poses = rearrange(ref_poses, 'b (t2 t3) p c -> (b t2) t3 p c', t2=t2, t3=t3)

        x = self.backbone(images)
        """
        x: [
            [(bs, 9), 48, H / 4=96, W / 4=72]    -> [(bs, 9), 256, 96, 72]  -> [(bs, 9), 256, 27, 17] -> [bs, 243, 17, 128]
            [(bs, 9), 96, H / 8=48, W / 8=36]    -> [(bs, 9), 256, 48, 36]  -> [(bs, 9), 256, 27, 17] -> [bs, 243, 17, 128]
            [(bs, 9), 192, H / 16=24, W / 16=18] -> [(bs, 9), 256, 24, 18]  -> [(bs, 9), 256, 27, 17] -> [bs, 243, 17, 128]
            [(bs, 9), 384, H / 32=12, W / 32=9]  -> [(bs, 9), 256, 12, 9]   -> [(bs, 9), 256, 27, 17] -> [bs, 243, 17, 128]
        ]
        """
        x = [self.feat_emb[i](f) for i, f in enumerate(x)]
        x = [F.grid_sample(f, ref_poses, align_corners=True) for f in x]
        x = [rearrange(f, '(b t2) c t3 p -> b (t2 t3) p c', t2=t2) for f in x]

        # x: [bs, 243, 17, 5, 128]
        x = torch.stack([x_0, *x], dim=-2)

        # [bs, 243, 17, 2] -> [bs, 16524, 1, 2]
        ref_points = repeat(ref_points, 'b t p c -> b (t p l) 1 c', l=self.n_levels)

        for decoder in self.decoders:
            x = decoder(x, ref_points)

        x = rearrange(x, 'b t p l c -> b t p (l c)')
        return x


class AdaptivePosePoolingv1(nn.Module):
    def __init__(self, enc_n_layers=16, enc_dim_in=3, enc_dim_feat=128, enc_dim_rep=512, enc_dim_out=3, enc_mlp_ratio=4., enc_act_layer=nn.GELU,
                 enc_attn_drop=0., enc_drop=0., enc_drop_path=0., enc_use_layer_scale=True, enc_layer_scale_init_value=1e-5, enc_use_adaptive_fusion=True,
                 enc_num_heads=8, enc_qkv_bias=False, enc_qkv_scale=None, enc_hierarchical=False, enc_use_temporal_similarity=True,
                 enc_temporal_connection_len=1, enc_use_tcn=False, enc_graph_only=False, enc_neighbour_num=2,
                 enc_pretrained='/home/xxxxxx/MotionAGFormer/checkpoint/motionagformer-b-h36m_hrnet.pth.tr',

                 dec_n_layers=4, dec_n_levels=4, dec_n_heads=8, dec_n_points=4, dec_drop_path=0.2,
                 dec_dim_img=128, dec_drop=.0, dec_mlp_ratio=4., dec_act_layer=nn.GELU,

                 backbone_feats=[32, 64, 128, 256],
                 backbone_type='hrnet', backbone_name='hrnet_w32_384_288',
                 backbone_pretrained='/home/xxxxxx/MotionAGFormer/data/pretrained/coco/pose_hrnet_w32_384x288.pth',
                 num_joints=17, n_frames=243):
        super(AdaptivePosePoolingv1, self).__init__()
        self.encoder = MotionAGTransformerEncoder(
            n_layers=enc_n_layers, dim_in=enc_dim_in, dim_feat=enc_dim_feat, dim_rep=enc_dim_rep,
            dim_out=enc_dim_out, mlp_ratio=enc_mlp_ratio, act_layer=enc_act_layer, attn_drop=enc_attn_drop,
            drop=enc_drop, drop_path=enc_drop_path, use_layer_scale=enc_use_layer_scale,
            layer_scale_init_value=enc_layer_scale_init_value, use_adaptive_fusion=enc_use_adaptive_fusion,
            num_heads=enc_num_heads, qkv_bias=enc_qkv_bias, qkv_scale=enc_qkv_scale, hierarchical=enc_hierarchical,
            use_temporal_similarity=enc_use_temporal_similarity, temporal_connection_len=enc_temporal_connection_len,
            use_tcn=enc_use_tcn, graph_only=enc_graph_only, neighbour_num=enc_neighbour_num, return_mid=True,
            num_joints=num_joints, n_frames=n_frames)

        self.decoder = MotionAGTransformerDecoder(
            n_layers=dec_n_layers, n_levels=dec_n_levels, n_heads=dec_n_heads, n_points=dec_n_points,
            drop_path=dec_drop_path, dim_img=dec_dim_img, drop=dec_drop, mlp_ratio=dec_mlp_ratio,
            act_layer=dec_act_layer, num_joints=num_joints, n_frames=n_frames,
            backbone_feats=backbone_feats, backbone_type=backbone_type,
            backbone_name=backbone_name, backbone_pretrained=backbone_pretrained)

        self.enc_pretrained = enc_pretrained
        if enc_pretrained is not None:
            checkpoint = torch.load(enc_pretrained)
            state_dict = {}
            for key in checkpoint['model'].keys():
                state_dict[key.replace('module.', '')] = checkpoint['model'][key].clone()
            self.encoder.load_state_dict(state_dict, strict=False)
            # set requires_grad = False
            for p in self.encoder.parameters():
                p.requires_grad = False

        self.proj = nn.Linear(enc_dim_feat, dec_dim_img)
        self.head = nn.Sequential(
            nn.LayerNorm(dec_dim_img*(dec_n_levels+1)),
            nn.Linear(dec_dim_img*(dec_n_levels+1), dec_dim_img),
            nn.Tanh(),
            nn.Linear(dec_dim_img, enc_dim_out)
            )

    def forward(self, images, poses_2d, poses_2d_crop):
        enc_output = self.encoder(poses_2d)
        enc_output = self.proj(enc_output)
        dec_output = self.decoder(images, enc_output, poses_2d_crop)

        dec_preds = self.head(dec_output)
        return dec_preds


def _test():
    from torchprofile import profile_macs
    import warnings
    warnings.filterwarnings('ignore')
    b, c, t, j = 1, 3, 243, 17
    poses = torch.randn((b, t, j, c)).cuda()
    images = torch.randn((b, 27, 3, 384, 288)).cuda()
    poses_crop = torch.randint(low=0, high=256, size=(b, t, j, c - 1)).float().cuda()

    model = AdaptivePosePoolingv1().cuda()
    model.eval()

    n_params = count_param_numbers(model) / 1000000
    print(f"Model parameter #: {n_params:.2f}M")
    print(f"Model MACs #: {profile_macs(model, (images, poses, poses_crop)):,}")

    # Warm-up to avoid timing fluctuations
    for _ in range(10):
        _ = model(images, poses, poses_crop)

    import time
    num_iterations = 100
    # Measure the inference time for 'num_iterations' iterations
    start_time = time.time()
    for _ in range(num_iterations):
        with torch.no_grad():
            _ = model(images, poses, poses_crop)
    end_time = time.time()

    # Calculate the average inference time per iteration
    average_inference_time = (end_time - start_time) / num_iterations

    # Calculate FPS
    fps = 1.0 / average_inference_time
    print(f"Time: {average_inference_time}")
    print(f"FPS: {fps}")

    out = model(images, poses, poses_crop)

    assert out.shape == (b, t, j, 3), f"Output shape should be {b}x{t}x{j}x3 but it is {out.shape}"


if __name__ == '__main__':
    _test()
