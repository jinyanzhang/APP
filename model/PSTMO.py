import torch
import torch.nn as nn
from model.modules.vanilla_transformer_encoder import Transformer
from model.modules.strided_transformer_encoder import Transformer as Transformer_reduce


class Linear(nn.Module):
    def __init__(self, linear_size, p_dropout=0.25):
        super(Linear, self).__init__()
        self.l_size = linear_size

        self.relu = nn.LeakyReLU(0.2, inplace=True)
        self.dropout = nn.Dropout(p_dropout)

        #self.w1 = nn.Linear(self.l_size, self.l_size)
        self.w1 = nn.Conv1d(self.l_size, self.l_size, kernel_size=1)
        self.batch_norm1 = nn.BatchNorm1d(self.l_size)

        #self.w2 = nn.Linear(self.l_size, self.l_size)
        self.w2 = nn.Conv1d(self.l_size, self.l_size, kernel_size=1)
        self.batch_norm2 = nn.BatchNorm1d(self.l_size)

    def forward(self, x):
        y = self.w1(x)
        y = self.batch_norm1(y)
        y = self.relu(y)
        y = self.dropout(y)

        y = self.w2(y)
        y = self.batch_norm2(y)
        y = self.relu(y)
        y = self.dropout(y)

        out = x + y

        return out


class FCBlock(nn.Module):

    def __init__(self, channel_in, channel_out, linear_size, block_num):
        super(FCBlock, self).__init__()

        self.linear_size = linear_size
        self.block_num = block_num
        self.layers = []
        self.channel_in = channel_in
        self.stage_num = 3
        self.p_dropout = 0.1
        #self.fc_1 = nn.Linear(self.channel_in, self.linear_size)
        self.fc_1 = nn.Conv1d(self.channel_in, self.linear_size, kernel_size=1)
        self.bn_1 = nn.BatchNorm1d(self.linear_size)
        for i in range(block_num):
            self.layers.append(Linear(self.linear_size, self.p_dropout))
        #self.fc_2 = nn.Linear(self.linear_size, channel_out)
        self.fc_2 = nn.Conv1d(self.linear_size, channel_out, kernel_size=1)

        self.layers = nn.ModuleList(self.layers)
        self.relu = nn.LeakyReLU(0.2, inplace=True)
        self.dropout = nn.Dropout(self.p_dropout)

    def forward(self, x):

        x = self.fc_1(x)
        x = self.bn_1(x)
        x = self.relu(x)
        x = self.dropout(x)
        for i in range(self.block_num):
            x = self.layers[i](x)
        x = self.fc_2(x)

        return x


class PSTMO(nn.Module):
    def __init__(self, args):
        super(PSTMO, self).__init__()

        layers, channel, d_hid, length  = args.layers, args.channel, args.d_hid, args.frames
        stride_num = args.stride_num
        self.num_joints_in, self.num_joints_out = args.n_joints, args.out_joints

        self.encoder = FCBlock(2*self.num_joints_in, channel, 2*channel, 1)

        self.Transformer = Transformer(layers, channel, d_hid, length=length)
        self.Transformer_reduce = Transformer_reduce(len(stride_num), channel, d_hid, \
            length=length, stride_num=stride_num)
        
        self.fcn = nn.Sequential(
            nn.BatchNorm1d(channel, momentum=0.1),
            nn.Conv1d(channel, 3*self.num_joints_out, kernel_size=1)
        )

        self.fcn_1 = nn.Sequential(
            nn.BatchNorm1d(channel, momentum=0.1),
            nn.Conv1d(channel, 3*self.num_joints_out, kernel_size=1)
        )

    def forward(self, x):
        x = x[:, :, :, :, 0].permute(0, 2, 3, 1).contiguous() 
        x_shape = x.shape

        x = x.view(x.shape[0], x.shape[1], -1) 
        x = x.permute(0, 2, 1).contiguous() 

        x = self.encoder(x) 

        x = x.permute(0, 2, 1).contiguous()
        x = self.Transformer(x) 

        x_VTE = x
        x_VTE = x_VTE.permute(0, 2, 1).contiguous()
        x_VTE = self.fcn_1(x_VTE) 

        x_VTE = x_VTE.view(x_shape[0], self.num_joints_out, -1, x_VTE.shape[2])
        x_VTE = x_VTE.permute(0, 2, 3, 1).contiguous().unsqueeze(dim=-1)

        x = self.Transformer_reduce(x) 
        x = x.permute(0, 2, 1).contiguous() 
        x = self.fcn(x) 

        x = x.view(x_shape[0], self.num_joints_out, -1, x.shape[2])
        x = x.permute(0, 2, 3, 1).contiguous().unsqueeze(dim=-1)
        
        return x, x_VTE


def _test():
    from torchprofile import profile_macs
    import warnings
    warnings.filterwarnings('ignore')
    b, c, t, j = 1, 3, 243, 17
    random_x = torch.randn((b, t, j, c)).cuda()

    model = PSTMO().cuda()
    
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