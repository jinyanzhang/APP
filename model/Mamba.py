import torch
from torch import nn
from einops import rearrange
from mamba_ssm import Mamba


class Permute(nn.Module):
    def __init__(self, *args):
        super().__init__()
        self.args = args

    def forward(self, x: torch.Tensor):
        return x.permute(*self.args)


class PatchEmbedding(nn.Module):
    def __init__(self, dim_in=3, d_model=384, frame_patch=9, patch_norm=True, num_joints=17, n_frames=243) -> None:
        super(PatchEmbedding, self).__init__()
        assert n_frames % frame_patch == 0
        self.dim_in = dim_in
        self.d_model = d_model
        self.frame_patch = frame_patch
        self.patch_norm = patch_norm
        self.num_joints = num_joints
        self.n_frames = n_frames  
        
        self.num_patches = n_frames // frame_patch
        
        self.patch_embed = self.__make_embedding()

    def __make_embedding(self):
        padding_1 = ((self.frame_patch - 1) // 2, (self.num_joints - 1) // 2)
        padding_2 = ((self.frame_patch // 3 - 1) // 2, (self.num_joints - 1) // 2)
        return nn.Sequential(
            # [bs, 3, 243, 17] -> # [bs, 243, 17, 3]
            Permute(0, 3, 1, 2),
            # [bs, 3, 243, 17] -> [bs, 64, 243, 17]
            nn.Conv2d(self.dim_in, self.d_model // 2, kernel_size=(self.frame_patch, self.num_joints), padding=padding_1),
            # [bs, 64, 243, 17] -> [bs, 243, 17, 64]
            (Permute(0, 2, 3, 1) if self.patch_norm else nn.Identity()),
            (nn.LayerNorm(self.d_model // 2) if self.patch_norm else nn.Identity()),
            # [bs, 243, 17, 64] -> [bs, 64, 243, 17]
            (Permute(0, 3, 1, 2) if self.patch_norm else nn.Identity()),
            nn.GELU(),
            # [bs, 64, 243, 17] -> [bs, 128, 243, 17]
            nn.Conv2d(self.d_model // 2, self.d_model, kernel_size=(self.frame_patch // 3, self.num_joints), padding=padding_2),
            # [bs, 128, 243, 17] -> [bs, 243, 17, 128]
            Permute(0, 2, 3, 1),
            (nn.LayerNorm(self.d_model) if self.patch_norm else nn.Identity()),
        )
        
    
    def forward(self, x):
        # x: [bs, 243, 17, 3] -> [bs, 243, 17, 384]
        x = self.patch_embed(x)
        b, *_, c = x.shape
        # x: [bs, 243, 17, 384] -> [bs, 27, 9, 17, 384]
        x = rearrange(x, 'b (n tp) p c -> b n tp p c', n=self.num_patches, tp=self.frame_patch)
        # rev_x: [bs, 27, 9, 17, 384]
        rev_x = torch.flip(x, dims=[1]).clone()
        # rev_x: [bs, 4131, 384]
        x = x.view(b, -1, c)
        # rev_x: [bs, 4131, 384]
        rev_x = rev_x.view(b, -1, c)
        # rev_x: [bs, 8262, 384]
        x = torch.cat((x, rev_x), dim=1)
        return x


class MambaPose(nn.Module):
    def __init__(self, dim_in=3, n_layers=12, d_model=384, frame_patch=9, patch_norm=True, d_state=16, d_conv=4, num_joints=17, n_frames=243) -> None:
        super(MambaPose, self).__init__()
        self.n_frames = n_frames
        self.num_joints = num_joints
        self.patch_embed = PatchEmbedding(dim_in, d_model, frame_patch, patch_norm, num_joints, n_frames)
        self.mamba_layers = nn.ModuleList([Mamba(d_model, d_state, d_conv) for _ in range(n_layers)])
        self.head = nn.Sequential(
            nn.Linear(d_model*2, d_model),
            nn.Tanh(),
            nn.Linear(d_model, 3)
        )
    
    def forward(self, x):
        x = self.patch_embed(x)
        for layer in self.mamba_layers:
            x = layer(x)
        x = rearrange(x, 'b (t p n) c -> b t p (n c)', t=self.n_frames, p=self.num_joints)
        x = self.head(x)
        return x


def _test():
    from torchprofile import profile_macs
    import warnings
    warnings.filterwarnings('ignore')
    b, c, t, j = 1, 3, 243, 17
    random_x = torch.randn((b, t, j, c)).cuda()

    model = MambaPose().cuda()
    model.eval()

    model_params = 0
    for parameter in model.parameters():
        model_params = model_params + parameter.numel()
    print(f"Model parameter #: {model_params / 1000000:,}M")
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

    assert out.shape == (b, t, j, 3), f"Output shape should be {b}x{t}x{j}x3 but it is {out.shape}"


if __name__ == '__main__':
    _test()
        