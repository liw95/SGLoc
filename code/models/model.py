import torch.nn as nn
import MinkowskiEngine as ME


class SGLoc(ME.MinkowskiNetwork):
    def __init__(self, in_channel=3, out_channel=3, D=3):
        super(SGLoc, self).__init__(D)
        conv_planes = [64, 128, 128, 256, 256, 512, 512, 512, 4096, 4096]
        self.CatChannels = conv_planes[1]
        self.conv1a = ConvBnReLU(in_channel, conv_planes[0], kernel_size=5)
        self.conv1b = ConvBnReLURes(conv_planes[0], conv_planes[1], stride=2, downsample_flag=True)
        self.conv2a = ConvBnReLURes(conv_planes[1], conv_planes[2], stride=2, downsample_flag=True)
        self.conv3a = ConvBnReLURes(conv_planes[2], conv_planes[3], downsample_flag=True)
        self.conv3b = ConvBnReLURes(conv_planes[3], conv_planes[4], stride=2, downsample_flag=True)
        self.conv4a = ConvBnReLURes(conv_planes[4], conv_planes[5], downsample_flag=True)
        self.conv4b = ConvBnReLURes(conv_planes[5], conv_planes[6])
        self.conv4c = ConvBnReLURes(conv_planes[6], conv_planes[7])
        self.conv5a = ConvBnReLURes(768, conv_planes[8], kernel_size=1, downsample_flag=True)
        self.conv5b = ConvBnReLURes(conv_planes[8], conv_planes[9], kernel_size=1)
        self.msf = attention_func(128, 256, 512, 128)
        self.convout = ConvFinal(conv_planes[9], out_channel)


    def weight_initialization(self):
        for m in self.modules():
            if isinstance(m, ME.MinkowskiConvolution):
                ME.utils.kaiming_normal_(m.kernel, mode="fan_out", nonlinearity="leaky_relu")

            if isinstance(m, ME.MinkowskiBatchNorm):
                nn.init.constant_(m.bn.weight, 1)
                nn.init.constant_(m.bn.bias, 0)

    def forward(self, x):

        out = self.conv1a(x)
        out1b = self.conv1b(out)    # 128
        out = self.conv2a(out1b)
        out3a = self.conv3a(out)  # 256
        out = self.conv3b(out3a)
        out = self.conv4a(out)
        out = self.conv4b(out)
        out = self.conv4c(out)
        # multi-scale
        out = self.conv5a(self.msf(out3a, out1b, out))

        out = self.conv5b(out)
        out = self.convout(out)

        return out


class Conv(nn.Module):
    def __init__(self,
                 inplanes,
                 planes,
                 kernel_size=3,
                 stride=2,
                 dilation=1,
                 dimension=3):
        super(Conv, self).__init__()
        self.net = ME.MinkowskiConvolution(inplanes,
                                    planes,
                                    kernel_size=kernel_size,
                                    stride=stride,
                                    dilation=dilation,
                                    dimension=dimension)

    def forward(self, x):
        return self.net(x)


class ConvBnReLU(nn.Module):
    def __init__(self,
                 inplanes,
                 planes,
                 kernel_size=3,
                 stride=1,
                 dilation=1,
                 dimension=3):
        super(ConvBnReLU, self).__init__()
        self.net = nn.Sequential(
            ME.MinkowskiConvolution(inplanes,
                                    planes,
                                    kernel_size=kernel_size,
                                    stride=stride,
                                    dilation=dilation,
                                    dimension=dimension),
            ME.MinkowskiBatchNorm(planes),
            ME.MinkowskiLeakyReLU(inplace=True))

    def forward(self, x):
        return self.net(x)


class ConvBnReLURes(nn.Module):
    def __init__(self,
                 inplanes,
                 planes,
                 stride=1,
                 kernel_size=3,
                 dilation=1,
                 downsample_flag=False,
                 dimension=3):
        super(ConvBnReLURes, self).__init__()
        self.conv1 = ME.MinkowskiConvolution(
            inplanes, planes, kernel_size=kernel_size, stride=stride, dilation=dilation, dimension=dimension)
        self.norm1 = ME.MinkowskiBatchNorm(planes)
        self.conv2 = ME.MinkowskiConvolution(
            planes, planes, kernel_size=kernel_size, stride=1, dilation=dilation, dimension=dimension)
        self.norm2 = ME.MinkowskiBatchNorm(planes)
        self.relu = ME.MinkowskiLeakyReLU(inplace=True)
        if downsample_flag:
            self.downsample = nn.Sequential(
                ME.MinkowskiConvolution(inplanes, planes, kernel_size=kernel_size, stride=stride, dilation=dilation,
                                        dimension=dimension),
                ME.MinkowskiBatchNorm(planes)
            )
        else:
            self.downsample = None

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.norm1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.norm2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class ConvFinal(nn.Module):
    def __init__(self,
                 inplanes,
                 planes,
                 kernel_size=1,
                 stride=1,
                 dilation=1,
                 dimension=3):
        super(ConvFinal, self).__init__()
        self.net = ME.MinkowskiConvolution(inplanes,
                                           planes,
                                           kernel_size=kernel_size,
                                           stride=stride,
                                           dilation=dilation,
                                           bias=True,
                                           dimension=dimension)

    def forward(self, x):
        return self.net(x)

class attention_func(nn.Module):
    def __init__(self, l_ch, s_ch, m_ch, cat_ch, D=3):
        super(attention_func, self).__init__()
        self.l_pool = ME.MinkowskiMaxPooling(kernel_size=4, stride=4, dimension=D)
        self.s_pool = ME.MinkowskiMaxPooling(kernel_size=2, stride=2, dimension=D)
        self.l_conv = ConvBnReLU(l_ch, cat_ch)
        self.s_conv = ConvBnReLU(s_ch, cat_ch)
        self.l_squeeze = ConvBnReLURes(cat_ch, 1, kernel_size=1, downsample_flag=True)
        self.s_squeeze = ConvBnReLURes(cat_ch, 1, kernel_size=1, downsample_flag=True)
        self.m_squeeze = ConvBnReLURes(m_ch, 1, kernel_size=1, downsample_flag=True)
        self.sigmoid = ME.MinkowskiSigmoid()
    def forward(self, s_feats, l_feats, m_feats):
        s_feats = self.s_conv(self.s_pool(s_feats))
        l_feats = self.l_conv(self.l_pool(l_feats))
        _s_feats = self.s_squeeze(s_feats)
        _l_feats = self.l_squeeze(l_feats)
        _m_feats = self.m_squeeze(m_feats)
        att_map_sum = self.sigmoid(_m_feats + _l_feats)
        att_map_mul = self.sigmoid(_m_feats + _s_feats)
        out_feats = ME.cat(m_feats, l_feats * att_map_sum, s_feats * att_map_mul)  # 512+128+128
        return out_feats