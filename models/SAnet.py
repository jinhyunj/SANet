import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from .torch_deform_conv.layers import ConvOffset2D


class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)


class ChannelGate(nn.Module):
    def __init__(self, gate_channels, reduction_ratio=16, pool_types=['avg', 'max']):
        super(ChannelGate, self).__init__()
        self.gate_channels = gate_channels
        self.mlp = nn.Sequential(
            Flatten(),
            nn.Linear(gate_channels, gate_channels // reduction_ratio),
            nn.ReLU(),
            nn.Linear(gate_channels // reduction_ratio, gate_channels)
        )
        self.pool_types = pool_types

    def forward(self, x):
        channel_att_sum = None
        for pool_type in self.pool_types:
            if pool_type=='avg':
                avg_pool = F.avg_pool2d(x, (x.size(2), x.size(3)), stride=(x.size(2), x.size(3)))
                channel_att_raw = self.mlp(avg_pool)
            elif pool_type=='max':
                max_pool = F.max_pool2d(x, (x.size(2), x.size(3)), stride=(x.size(2), x.size(3)))
                channel_att_raw = self.mlp(max_pool)

            if channel_att_sum is None:
                channel_att_sum = channel_att_raw
            else:
                channel_att_sum = channel_att_sum + channel_att_raw

        scale = F.sigmoid(channel_att_sum).unsqueeze(2).unsqueeze(3)
        return scale


class CBAM(nn.Module):
    def __init__(self, gate_channels, reduction_ratio=16, pool_types=['avg', 'max']):
        super(CBAM, self).__init__()
        self.ChannelGate = ChannelGate(gate_channels, reduction_ratio, pool_types)

    def forward(self, x):
        return self.ChannelGate(x)


class ResnetBlock(nn.Module):
    def __init__(self, dim, padding_type='reflect', norm_layer=nn.InstanceNorm2d, use_dropout=False):
        super(ResnetBlock, self).__init__()
        self.conv_block = self.build_conv_block(dim, padding_type, norm_layer, use_dropout)

    def build_conv_block(self, dim, padding_type, norm_layer, use_dropout):
        conv_block = []
        p = 0
        if padding_type == 'reflect':
            conv_block += [nn.ReflectionPad2d(1)]
        elif padding_type == 'replicate':
            conv_block += [nn.ReplicationPad2d(1)]
        elif padding_type == 'zero':
            p = 1
        else:
            raise NotImplementedError('padding [%s] is not implemented' % padding_type)

        conv_block += [nn.Conv2d(dim, dim, kernel_size=3, padding=p), norm_layer(dim), nn.ReLU(True)]
        if use_dropout:
            conv_block += [nn.Dropout(0.5)]

        p = 0
        if padding_type == 'reflect':
            conv_block += [nn.ReflectionPad2d(1)]
        elif padding_type == 'replicate':
            conv_block += [nn.ReplicationPad2d(1)]
        elif padding_type == 'zero':
            p = 1
        else:
            raise NotImplementedError('padding [%s] is not implemented' % padding_type)
        conv_block += [nn.Conv2d(dim, dim, kernel_size=3, padding=p), norm_layer(dim)]

        return nn.Sequential(*conv_block)

    def forward(self, x):
        out = x + self.conv_block(x)  # add skip connections
        return out


class Encoder(nn.Module):
    def __init__(self, ngf=64, n_blocks=4):
        super(Encoder, self).__init__()
        model = [nn.ReflectionPad2d(3),
                 nn.Conv2d(3, ngf, kernel_size=7, padding=0),
                 nn.InstanceNorm2d(ngf),
                 nn.ReLU(True)]

        n_downsampling = 2
        for i in range(n_downsampling):
            mult = 2**i
            model += [nn.Conv2d(ngf*mult, ngf * mult * 2, kernel_size=3,
                                stride=2, padding=1),
                      nn.InstanceNorm2d(ngf * mult * 2),
                      nn.ReLU(True)]
        self.model = nn.Sequential(*model)

        mult = 2**n_downsampling
        resnet = [nn.Conv2d(ngf*mult, ngf * mult, kernel_size=3,
                            stride=1, padding=1),
                  nn.InstanceNorm2d(ngf * mult),
                  nn.ReLU(True),
                  nn.Conv2d(ngf * mult, ngf * mult, kernel_size=3,
                            stride=1, padding=1),
                  nn.InstanceNorm2d(ngf * mult),
                  nn.ReLU(True)
                  ]

        for i in range(n_blocks):
            resnet += [ResnetBlock(ngf * mult, use_dropout=False)]
        self.resnet = nn.Sequential(*resnet)

    def forward(self, x):
        x = self.model(x)
        return self.resnet(x)


class Trans(nn.Module):
    def __init__(self):
        super(Trans, self).__init__()

        def warp():
            return nn.Sequential(
                ConvOffset2D(256),
                nn.Conv2d(256, 256, 3, 1, 1),
                nn.ReLU(True),
                nn.InstanceNorm2d(256)
            )

        self.cbam_1 = CBAM(256, 256)
        self.cbam_2 = CBAM(256, 256)
        self.cbam_3 = CBAM(256, 256)
        self.cbam_4 = CBAM(256, 256)

        self.warp_1 = warp()
        self.warp_2 = warp()
        self.warp_3 = warp()
        self.warp_4 = warp()

        h = 64
        height = 32
        width = 128
        i = torch.arange(0, height)
        j = torch.arange(0, width)
        ii, jj = torch.meshgrid(i, j)

        x = h / 2. - h / 2. / height * (height - 1 - ii) * torch.sin(2 * np.pi * jj / width)
        y = h / 2. + h / 2. / height * (height - 1 - ii) * torch.cos(2 * np.pi * jj / width)

        grid = torch.stack((x, y), 2)
        grid = grid.unsqueeze(0)
        factor = torch.FloatTensor([[[[2 / h, 2 / h]]]])
        grid = grid * factor - 1
        self.grid = grid.cuda()
        self.softmax = nn.Softmax(dim=3)

    def forward(self, x):
        b, _,_,_ = x.size()
        grid = self.grid.repeat(b, 1, 1, 1)
        polar_AS = F.grid_sample(x, grid)
        scale_1 = self.cbam_1(polar_AS)
        scale_2 = self.cbam_2(polar_AS)
        scale_3 = self.cbam_3(polar_AS)
        scale_4 = self.cbam_4(polar_AS)
        scale_tot = torch.cat((scale_1, scale_2, scale_3, scale_4), 3)
        new_scale = self.softmax(scale_tot)

        feat_AS_1 = polar_AS*(new_scale[:,:,:,0].unsqueeze(3).expand_as(polar_AS))
        feat_AS_2 = polar_AS*(new_scale[:,:,:,1].unsqueeze(3).expand_as(polar_AS))
        feat_AS_3 = polar_AS*(new_scale[:,:,:,2].unsqueeze(3).expand_as(polar_AS))
        feat_AS_4 = polar_AS*(new_scale[:,:,:,3].unsqueeze(3).expand_as(polar_AS))

        return self.warp_1(feat_AS_1), self.warp_2(feat_AS_2), self.warp_3(feat_AS_3), self.warp_4(feat_AS_4)


class Decoder(nn.Module):
    def __init__(self, ngf=64, n_blocks=5):
        assert (n_blocks >= 0)
        super(Decoder, self).__init__()
        n_downsampling = 2
        mult = 2 ** n_downsampling
        model = [ResnetBlock(ngf * mult, use_dropout=True)]

        for i in range(n_blocks-1):
            model += [ResnetBlock(ngf * mult, use_dropout=True)]

        for i in range(n_downsampling):
            mult = 2 ** (n_downsampling - i)
            model += [nn.Upsample(scale_factor=2),
                      nn.Conv2d(ngf*mult, int(ngf*mult/2), kernel_size=3,
                                stride=1, padding=1),
                      nn.InstanceNorm2d(int(ngf * mult / 2)),
                      nn.ReLU(True)]
        model += [nn.ReflectionPad2d(3),
                  nn.Conv2d(ngf, 3, kernel_size=7, padding=0),
                  nn.Tanh()]

        self.model = nn.Sequential(*model)

    def forward(self, x):
        return self.model(x)
