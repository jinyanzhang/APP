from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import math
import torch
from torch import nn
from model.backbones.resnet import *
from einops import reduce


__all__ = ['CPN50', 'CPN101']


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * 2, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 2)
        self.relu = nn.ReLU(inplace=True)

        self.downsample = nn.Sequential(
            nn.Conv2d(inplanes, planes * 2,
                      kernel_size=1, stride=stride, bias=False),
            nn.BatchNorm2d(planes * 2),
        )

        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class GlobalNet(nn.Module):
    def __init__(self, channel_settings, output_shape, num_class):
        super(GlobalNet, self).__init__()
        self.channel_settings = channel_settings
        laterals, upsamples, predict = [], [], []
        for i in range(len(channel_settings)):
            laterals.append(self._lateral(channel_settings[i]))
            predict.append(self._predict(output_shape, num_class))
            if i != len(channel_settings) - 1:
                upsamples.append(self._upsample())
        self.laterals = nn.ModuleList(laterals)
        self.upsamples = nn.ModuleList(upsamples)
        self.predict = nn.ModuleList(predict)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _lateral(self, input_size):
        layers = []
        layers.append(nn.Conv2d(input_size, 256,
            kernel_size=1, stride=1, bias=False))
        layers.append(nn.BatchNorm2d(256))
        layers.append(nn.ReLU(inplace=True))

        return nn.Sequential(*layers)

    def _upsample(self):
        layers = []
        layers.append(torch.nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True))
        layers.append(torch.nn.Conv2d(256, 256,
            kernel_size=1, stride=1, bias=False))
        layers.append(nn.BatchNorm2d(256))

        return nn.Sequential(*layers)

    def _predict(self, output_shape, num_class):
        layers = []
        layers.append(nn.Conv2d(256, 256,
            kernel_size=1, stride=1, bias=False))
        layers.append(nn.BatchNorm2d(256))
        layers.append(nn.ReLU(inplace=True))

        layers.append(nn.Conv2d(256, num_class,
            kernel_size=3, stride=1, padding=1, bias=False))
        layers.append(nn.Upsample(size=output_shape, mode='bilinear', align_corners=True))
        layers.append(nn.BatchNorm2d(num_class))

        return nn.Sequential(*layers)

    def forward(self, x):
        global_fms, global_outs = [], []
        for i in range(len(self.channel_settings)):
            if i == 0:
                feature = self.laterals[i](x[i])
            else:
                feature = self.laterals[i](x[i]) + up
            global_fms.append(feature)
            if i != len(self.channel_settings) - 1:
                up = self.upsamples[i](feature)
            feature = self.predict[i](feature)
            global_outs.append(feature)

        return global_fms, global_outs


class RefineNet(nn.Module):
    def __init__(self, lateral_channel, out_shape, num_class):
        super(RefineNet, self).__init__()
        cascade = []
        num_cascade = 4
        for i in range(num_cascade):
            cascade.append(self._make_layer(lateral_channel, num_cascade - i - 1, out_shape))
        self.cascade = nn.ModuleList(cascade)
        # self.final_predict = self._predict(4 * lateral_channel, num_class)

    def _make_layer(self, input_channel, num, output_shape):
        layers = []
        for i in range(num):
            layers.append(Bottleneck(input_channel, 128))
        # layers.append(nn.Upsample(size=output_shape, mode='bilinear', align_corners=True))
        return nn.Sequential(*layers)

    def _predict(self, input_channel, num_class):
        layers = []
        layers.append(Bottleneck(input_channel, 128))
        layers.append(nn.Conv2d(256, num_class,
                                kernel_size=3, stride=1, padding=1, bias=False))
        layers.append(nn.BatchNorm2d(num_class))
        return nn.Sequential(*layers)

    def forward(self, x):
        refine_fms = []
        for i in range(4):
            refine_fm = self.cascade[i](x[i])
            refine_fm = reduce(refine_fm, 'b (c k) h w -> b c h w', 'mean', k=2**i)
            refine_fms.append(refine_fm)
        return refine_fms
        # out = torch.cat(refine_fms, dim=1)
        # out = self.final_predict(out)
        # return out


class CPN(nn.Module):
    def __init__(self, resnet, output_shape, num_class):
        super(CPN, self).__init__()
        channel_settings = [2048, 1024, 512, 256]
        self.resnet = resnet
        self.global_net = GlobalNet(channel_settings, output_shape, num_class)
        self.refine_net = RefineNet(channel_settings[-1], output_shape, num_class)

    def forward(self, x):
        res_out = self.resnet(x)
        global_fms, global_outs = self.global_net(res_out)
        refine_out = self.refine_net(global_fms)

        # return global_outs, refine_out
        return refine_out[::-1]


def build_model(model_name='cpn_50_384_288'):
    if model_name not in [
        'cpn_50_256_192',
        'cpn_50_384_288',
        'cpn_101_384_288'
    ]:
        raise ValueError(f'Got model_name {model_name}, expected cpn_50/101_256/384_192/288')
    name_splits = model_name.split('_')
    tmp_model_type = name_splits[1]
    tmp_heatmap_size = (int(name_splits[2]) // 4, int(name_splits[3]) // 4)
    if tmp_model_type == '50':
        res50 = resnet50(pretrained=True)
        model = CPN(res50, tmp_heatmap_size, 17)
    else:
        res101 = resnet101(pretrained=True)
        model = CPN(res101, tmp_heatmap_size, 17)
    return model


def load_model(model, pretrained_model):
    if pretrained_model is not None:
        if os.path.isfile(pretrained_model):
            checkpoint = torch.load(pretrained_model)
            state_dict = {}
            for key in checkpoint['state_dict'].keys():
                state_dict[key.replace('module.', '')] = checkpoint['state_dict'][key]
            ret = model.load_state_dict(state_dict, strict=False)
            print(f'Successfully read pretrained model from {pretrained_model}')
            print(ret)
    return model


if __name__ == '__main__':
    model_type = 50
    input_shape = [256, 192]

    model = build_model(f'cpn_{model_type}_{input_shape[0]}_{input_shape[1]}')
    model = load_model(model, f'/home/xxxxxx/MotionAGFormer/data/pretrained/CPN{model_type}_{input_shape[0]}x{input_shape[1]}.pth.tar').cuda()
    images = torch.randn((16, 3, input_shape[0], input_shape[1]), dtype=torch.float32).cuda()

    out = model(images)
    pass
