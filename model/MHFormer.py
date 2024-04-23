import torch
import torch.nn as nn
from einops import rearrange

import sys
sys.path.append('/home/xxxxxx/MotionAGFormer')

from model.modules.trans import Transformer as Transformer_encoder
from model.modules.trans_hypothesis import Transformer as Transformer_hypothesis


class MHFormer(nn.Module):
    def __init__(self, dim_in=2, dim_feat=1024, channel=512, layers=3, num_joints=17, n_frames=243, return_mid=False):
        super(MHFormer, self).__init__()
        self.dim_in = dim_in
        self.return_mid = return_mid

        ## MHG
        self.norm_1 = nn.LayerNorm(n_frames)
        self.norm_2 = nn.LayerNorm(n_frames)
        self.norm_3 = nn.LayerNorm(n_frames)

        self.Transformer_encoder_1 = Transformer_encoder(4, n_frames, n_frames*2, length=2*num_joints, h=9)
        self.Transformer_encoder_2 = Transformer_encoder(4, n_frames, n_frames*2, length=2*num_joints, h=9)
        self.Transformer_encoder_3 = Transformer_encoder(4, n_frames, n_frames*2, length=2*num_joints, h=9)

        ## Embedding
        if n_frames > 27:
            self.embedding_1 = nn.Conv1d(2*num_joints, channel, kernel_size=1)
            self.embedding_2 = nn.Conv1d(2*num_joints, channel, kernel_size=1)
            self.embedding_3 = nn.Conv1d(2*num_joints, channel, kernel_size=1)
        else:
            self.embedding_1 = nn.Sequential(
                nn.Conv1d(2*num_joints, channel, kernel_size=1),
                nn.BatchNorm1d(channel, momentum=0.1),
                nn.ReLU(inplace=True),
                nn.Dropout(0.25)
            )

            self.embedding_2 = nn.Sequential(
                nn.Conv1d(2*num_joints, channel, kernel_size=1),
                nn.BatchNorm1d(channel, momentum=0.1),
                nn.ReLU(inplace=True),
                nn.Dropout(0.25)
            )

            self.embedding_3 = nn.Sequential(
                nn.Conv1d(2*num_joints, channel, kernel_size=1),
                nn.BatchNorm1d(channel, momentum=0.1),
                nn.ReLU(inplace=True),
                nn.Dropout(0.25)
            )

        ## SHR & CHI
        self.Transformer_hypothesis = Transformer_hypothesis(layers, channel, dim_feat, length=n_frames)
        
        ## Regression
        self.regression = nn.Sequential(
            nn.BatchNorm1d(channel*3, momentum=0.1),
            nn.Conv1d(channel*3, 3*num_joints, kernel_size=1)
        )

    def forward(self, x):
        x = x[..., :self.dim_in]
        B, F, J, C = x.shape
        x = rearrange(x, 'b f j c -> b (j c) f').contiguous()

        ## MHG
        x_1 = x   + self.Transformer_encoder_1(self.norm_1(x))
        x_2 = x_1 + self.Transformer_encoder_2(self.norm_2(x_1)) 
        x_3 = x_2 + self.Transformer_encoder_3(self.norm_3(x_2))
        
        ## Embedding
        x_1 = self.embedding_1(x_1).permute(0, 2, 1).contiguous() 
        x_2 = self.embedding_2(x_2).permute(0, 2, 1).contiguous()
        x_3 = self.embedding_3(x_3).permute(0, 2, 1).contiguous()

        ## SHR & CHI
        x = self.Transformer_hypothesis(x_1, x_2, x_3) 

        ## Regression
        x = x.permute(0, 2, 1).contiguous()
        
        if self.return_mid:
            # x = rearrange()
            return x
        
        x = self.regression(x)
        x = rearrange(x, 'b (j c) f -> b f j c', j=J).contiguous()

        return x


def _test():
    from torchprofile import profile_macs
    import warnings
    warnings.filterwarnings('ignore')
    b, c, t, j = 1, 3, 243, 17
    random_x = torch.randn((b, t, j, c)).cuda()

    model = MHFormer(return_mid=True).cuda()
    
    # state_dict = torch.load('/data-home/checkpoints/model_117_3975.pth')
    # ret = model.load_state_dict(state_dict, strict=True)
    # print(ret)
    
    # checkpoint = {'model': state_dict}
    # torch.save(checkpoint, '/data-home/checkpoints/hot-h36m-sh-author.pth.tr')
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

    assert out.shape == (b, t, j, 3), f"Output shape should be {b}x{t}x{j}x3 but it is {out.shape}"


if __name__ == '__main__':
    _test()