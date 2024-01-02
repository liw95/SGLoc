import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
import MinkowskiEngine as ME

class BasicBlock(nn.Module):
    def __init__(self,
                 inplanes,
                 planes,
                 stride=1,
                 dilation=1,
                 bn_momentum=0.1,
                 dimension=3):
        super(BasicBlock, self).__init__()
        self.conv1 = ME.MinkowskiConvolution(
            inplanes, planes, kernel_size=3, stride=stride, dilation=dilation, dimension=dimension)
        self.norm1 = ME.MinkowskiBatchNorm(planes, momentum=bn_momentum)
        self.conv2 = ME.MinkowskiConvolution(
            planes, planes, kernel_size=3, stride=1, dilation=dilation, dimension=dimension)
        self.norm2 = ME.MinkowskiBatchNorm(planes, momentum=bn_momentum)
        self.relu = ME.MinkowskiReLU(inplace=True)
        # self.downsample = nn.Sequential(
        #     ME.MinkowskiConvolution(inplanes, planes, kernel_size=1, stride=stride, dilation=dilation,
        #                             dimension=dimension),
        #     ME.MinkowskiBatchNorm(planes)
        # )

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.norm1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.norm2(out)

        # if self.downsample is not None:
        #     residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out

class Pooling(nn.Module):
    def __init__(self,
                 inplanes,
                 planes,
                 stride=2,
                 dilation=1,
                 bn_momentum=0.1,
                 dimension=3):
        super(Pooling, self).__init__()
        self.pooling = ME.MinkowskiConvolution(inplanes,
                                               planes,
                                               kernel_size=3,
                                               stride=stride,
                                               dilation=dilation,
                                               dimension=dimension)
        self.norm    = ME.MinkowskiBatchNorm(planes, momentum=bn_momentum)

    def forward(self, x):
        out = self.pooling(x)
        out = self.norm(out)

        return out

class UpConv(nn.Module):
    def __init__(self,
                 inplanes,
                 planes,
                 kernel_size=5,
                 stride=1,
                 dilation=1,
                 bn_momentum=0.1,
                 dimension=3):
        super(UpConv, self).__init__()
        self.conv = ME.MinkowskiConvolution(inplanes,
                                            planes,
                                            kernel_size=kernel_size,
                                            stride=stride,
                                            dilation=dilation,
                                            dimension=dimension)
        self.norm  = ME.MinkowskiBatchNorm(planes, momentum=bn_momentum)

    def forward(self, x):
        out = self.conv(x)
        out = self.norm(out)

        return out

class ConvTrasnpose(nn.Module):
    def __init__(self,
                 inplanes,
                 planes,
                 kernel_size=3,
                 stride=2,
                 dilation=1,
                 bn_momentum=0.1,
                 dimension=3):
        super(ConvTrasnpose, self).__init__()
        self.conv_tr = ME.MinkowskiConvolutionTranspose(inplanes,
                                                        planes,
                                                        kernel_size=kernel_size,
                                                        stride=stride,
                                                        dilation=dilation,
                                                        dimension=dimension)
        self.norm    = ME.MinkowskiBatchNorm(planes, momentum=bn_momentum)

    def forward(self, x):
        out = self.conv_tr(x)
        out = self.norm(out)

        return out