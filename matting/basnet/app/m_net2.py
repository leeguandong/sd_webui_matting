from collections import OrderedDict
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils import model_zoo
from torchvision.models.densenet import densenet121, densenet161
from torchvision.models.squeezenet import squeezenet1_1


def load_weights_sequential(target, source_state):
    """
    new_dict = OrderedDict()
    for (k1, v1), (k2, v2) in zip(target.state_dict().items(), source_state.items()):
        new_dict[k1] = v2
    """
    model_to_load = {k: v for k, v in source_state.items() if k in target.state_dict().keys()}
    target.load_state_dict(model_to_load)


'''
    Implementation of dilated ResNet-101 with deep supervision. Downsampling is changed to 8x
'''
model_urls = {
    'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
    'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
    'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
    'resnet152': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth',
}


def conv3x3(in_planes, out_planes, stride=1, dilation=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=dilation, dilation=dilation, bias=False)


class bottle(nn.Module):
    """
    Convolution Block
    """

    def __init__(self, in_ch, out_ch, k=1, s=1, p=0):
        super(bottle, self).__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=k, stride=s, padding=p, bias=True),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True))

    def forward(self, x):
        x = self.conv(x)
        return x


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, dilation=1):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride=stride, dilation=dilation)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes, stride=1, dilation=dilation)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None, dilation=1):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, dilation=dilation,
                               padding=dilation, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 4)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
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


class ResNet(nn.Module):
    def __init__(self, block, layers=(3, 4, 23, 3)):
        self.inplanes = 64
        super(ResNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=1, dilation=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=1, dilation=4)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, stride=1, dilation=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = [block(self.inplanes, planes, stride, downsample)]
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, dilation=dilation))

        return nn.Sequential(*layers)

    def forward(self, x):
        x1 = self.conv1(x)
        print('x1:', x1.size())
        x2 = self.bn1(x1)
        x3 = self.relu(x2)
        x4 = self.maxpool(x3)
        print('x4:', x4.size())
        x5 = self.layer1(x4)

        x6 = self.layer2(x5)
        print('x6:', x6.size())
        x7 = self.layer3(x6)
        x8 = self.layer4(x7)
        print('x8:', x8.size())

        return x1, x4, x6, x8


'''
    Implementation of DenseNet with deep supervision. Downsampling is changed to 8x 
'''


class _DenseLayer(nn.Sequential):
    def __init__(self, num_input_features, growth_rate, bn_size, drop_rate):
        super(_DenseLayer, self).__init__()
        self.add_module('norm.1', nn.BatchNorm2d(num_input_features)),
        self.add_module('relu.1', nn.ReLU(inplace=True)),
        self.add_module('conv.1', nn.Conv2d(num_input_features, bn_size *
                                            growth_rate, kernel_size=1, stride=1, bias=False)),
        self.add_module('norm.2', nn.BatchNorm2d(bn_size * growth_rate)),
        self.add_module('relu.2', nn.ReLU(inplace=True)),
        self.add_module('conv.2', nn.Conv2d(bn_size * growth_rate, growth_rate,
                                            kernel_size=3, stride=1, padding=1, bias=False)),
        self.drop_rate = drop_rate

    def forward(self, x):
        new_features = super(_DenseLayer, self).forward(x)
        if self.drop_rate > 0:
            new_features = F.dropout(new_features, p=self.drop_rate, training=self.training)
        return torch.cat([x, new_features], 1)


class _DenseBlock(nn.Sequential):
    def __init__(self, num_layers, num_input_features, bn_size, growth_rate, drop_rate):
        super(_DenseBlock, self).__init__()
        for i in range(num_layers):
            layer = _DenseLayer(num_input_features + i * growth_rate, growth_rate, bn_size, drop_rate)
            self.add_module('denselayer%d' % (i + 1), layer)


class _Transition(nn.Sequential):
    def __init__(self, num_input_features, num_output_features, downsample=True):
        super(_Transition, self).__init__()
        self.add_module('norm', nn.BatchNorm2d(num_input_features))
        self.add_module('relu', nn.ReLU(inplace=True))
        self.add_module('conv', nn.Conv2d(num_input_features, num_output_features,
                                          kernel_size=1, stride=1, bias=False))
        if downsample:
            self.add_module('pool', nn.AvgPool2d(kernel_size=2, stride=2))
        else:
            self.add_module('pool', nn.AvgPool2d(kernel_size=1, stride=1))  # compatibility hack


class DenseNet(nn.Module):
    def __init__(self, growth_rate=32, block_config=(6, 12, 24, 16),
                 num_init_features=64, bn_size=4, drop_rate=0, pretrained=True):

        super(DenseNet, self).__init__()

        # First convolution
        self.start_features = nn.Sequential(OrderedDict([
            ('conv0', nn.Conv2d(3, num_init_features, kernel_size=7, stride=2, padding=3, bias=False)),
            ('norm0', nn.BatchNorm2d(num_init_features)),
            ('relu0', nn.ReLU(inplace=True)),
            ('pool0', nn.MaxPool2d(kernel_size=3, stride=2, padding=1)),
        ]))

        # Each denseblock
        num_features = num_init_features

        init_weights = list(densenet121(pretrained=True).features.children())
        start = 0
        for i, c in enumerate(self.start_features.children()):
            if pretrained:
                c.load_state_dict(init_weights[i].state_dict())
            start += 1
        self.blocks = nn.ModuleList()
        for i, num_layers in enumerate(block_config):
            block = _DenseBlock(num_layers=num_layers, num_input_features=num_features,
                                bn_size=bn_size, growth_rate=growth_rate, drop_rate=drop_rate)
            if pretrained:
                block.load_state_dict(init_weights[start].state_dict())
            start += 1
            self.blocks.append(block)
            setattr(self, 'denseblock%d' % (i + 1), block)

            num_features = num_features + num_layers * growth_rate
            if i != len(block_config) - 1:
                downsample = i < 1
                trans = _Transition(num_input_features=num_features, num_output_features=num_features // 2,
                                    downsample=downsample)
                if pretrained:
                    trans.load_state_dict(init_weights[start].state_dict())
                start += 1
                self.blocks.append(trans)
                setattr(self, 'transition%d' % (i + 1), trans)
                num_features = num_features // 2

    def forward(self, x):
        out = self.start_features(x)
        deep_features = None
        for i, block in enumerate(self.blocks):
            out = block(out)
            if i == 5:
                deep_features = out

        return out, deep_features


class Fire(nn.Module):

    def __init__(self, inplanes, squeeze_planes,
                 expand1x1_planes, expand3x3_planes, dilation=1):
        super(Fire, self).__init__()
        self.inplanes = inplanes
        self.squeeze = nn.Conv2d(inplanes, squeeze_planes, kernel_size=1)
        self.squeeze_activation = nn.ReLU(inplace=True)
        self.expand1x1 = nn.Conv2d(squeeze_planes, expand1x1_planes,
                                   kernel_size=1)
        self.expand1x1_activation = nn.ReLU(inplace=True)
        self.expand3x3 = nn.Conv2d(squeeze_planes, expand3x3_planes,
                                   kernel_size=3, padding=dilation, dilation=dilation)
        self.expand3x3_activation = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.squeeze_activation(self.squeeze(x))
        return torch.cat([
            self.expand1x1_activation(self.expand1x1(x)),
            self.expand3x3_activation(self.expand3x3(x))
        ], 1)


class SqueezeNet(nn.Module):

    def __init__(self, pretrained=False):
        super(SqueezeNet, self).__init__()

        self.feat_1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True)
        )
        self.feat_2 = nn.Sequential(
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            Fire(64, 16, 64, 64),
            Fire(128, 16, 64, 64)
        )
        self.feat_3 = nn.Sequential(
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            Fire(128, 32, 128, 128, 2),
            Fire(256, 32, 128, 128, 2)
        )
        self.feat_4 = nn.Sequential(
            Fire(256, 48, 192, 192, 4),
            Fire(384, 48, 192, 192, 4),
            Fire(384, 64, 256, 256, 4),
            Fire(512, 64, 256, 256, 4)
        )
        if pretrained:
            weights = squeezenet1_1(pretrained=True).features.state_dict()
            load_weights_sequential(self, weights)

    def forward(self, x):
        f1 = self.feat_1(x)
        f2 = self.feat_2(f1)
        f3 = self.feat_3(f2)
        f4 = self.feat_4(f3)
        return f4, f3


'''
    Handy methods for construction
'''


class basenet(object):

    def squeezenet(pretrained=True):
        return SqueezeNet(pretrained)

    def densenet(pretrained=True):
        return DenseNet(pretrained=pretrained)

    def resnet18(pretrained=True):
        model = ResNet(BasicBlock, [2, 2, 2, 2])
        if pretrained:
            load_weights_sequential(model, model_zoo.load_url(model_urls['resnet18']))
        return model

    def resnet34(pretrained=True):
        model = ResNet(BasicBlock, [3, 4, 6, 3])
        if pretrained:
            load_weights_sequential(model, model_zoo.load_url(model_urls['resnet34']))
        return model

    def resnet50(pretrained=True):
        model = ResNet(Bottleneck, [3, 4, 6, 3])
        if pretrained:
            load_weights_sequential(model, model_zoo.load_url(model_urls['resnet50']))
        return model

    def resnet101(pretrained=True):
        model = ResNet(Bottleneck, [3, 4, 23, 3])
        if pretrained:
            load_weights_sequential(model, model_zoo.load_url(model_urls['resnet101']))
        return model

    def resnet152(pretrained=True):
        model = ResNet(Bottleneck, [3, 8, 36, 3])
        if pretrained:
            load_weights_sequential(model, model_zoo.load_url(model_urls['resnet152']))
        return model


class PSPModule(nn.Module):
    def __init__(self, features, out_features=1024, sizes=(1, 2, 3, 6)):
        super().__init__()
        self.stages = []
        self.stages = nn.ModuleList([self._make_stage(features, size) for size in sizes])
        self.bottleneck = nn.Conv2d(features * (len(sizes) + 1), out_features, kernel_size=1)
        self.relu = nn.ReLU()

    def _make_stage(self, features, size):
        prior = nn.AdaptiveAvgPool2d(output_size=(size, size))
        conv = nn.Conv2d(features, features, kernel_size=1, bias=False)
        return nn.Sequential(prior, conv)

    def forward(self, feats):
        h, w = feats.size(2), feats.size(3)
        priors = [F.interpolate(input=stage(feats), size=(h, w), mode='bilinear', align_corners=True) for stage in
                  self.stages] + [feats]
        bottle = self.bottleneck(torch.cat(priors, 1))
        return self.relu(bottle)


class PSPUpsample(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.PReLU()
        )

    def forward(self, x):
        h, w = 2 * x.size(2), 2 * x.size(3)
        p = F.interpolate(input=x, size=(h, w), mode='bilinear', align_corners=True)
        return self.conv(p)


class Net(nn.Module):
    def __init__(self, n_classes=1):
        super().__init__()

        # -----------------------------------------------------------------
        # alpha encoder
        # ---------------------
        # stage-1
        self.conv_1_1 = nn.Sequential(nn.Conv2d(4, 32, 3, 1, 1, bias=True), nn.BatchNorm2d(32), nn.ReLU())
        self.conv_1_2 = nn.Sequential(nn.Conv2d(32, 32, 3, 1, 1, bias=True), nn.BatchNorm2d(32), nn.ReLU())
        self.max_pooling_1 = nn.MaxPool2d(kernel_size=2, stride=2, return_indices=True)

        # stage-2
        self.conv_2_1 = nn.Sequential(nn.Conv2d(32, 64, 3, 1, 1, bias=True), nn.BatchNorm2d(64), nn.ReLU())
        self.conv_2_2 = nn.Sequential(nn.Conv2d(64, 64, 3, 1, 1, bias=True), nn.BatchNorm2d(64), nn.ReLU())
        self.max_pooling_2 = nn.MaxPool2d(kernel_size=2, stride=2, return_indices=True)

        # stage-3
        self.conv_3_1 = nn.Sequential(nn.Conv2d(64, 128, 3, 1, 1, bias=True), nn.BatchNorm2d(128), nn.ReLU())
        self.conv_3_2 = nn.Sequential(nn.Conv2d(128, 128, 3, 1, 1, bias=True), nn.BatchNorm2d(128), nn.ReLU())
        # self.conv_3_3 = nn.Sequential(nn.Conv2d(256, 256, 3, 1, 1, bias=True), nn.BatchNorm2d(256), nn.ReLU())
        self.max_pooling_3 = nn.MaxPool2d(kernel_size=2, stride=2, return_indices=True)

        # stage-4
        self.conv_4_1 = nn.Sequential(nn.Conv2d(128, 256, 3, 1, 1, bias=True), nn.BatchNorm2d(256), nn.ReLU())
        self.conv_4_2 = nn.Sequential(nn.Conv2d(256, 256, 3, 1, 1, bias=True), nn.BatchNorm2d(256), nn.ReLU())
        # self.conv_4_3 = nn.Sequential(nn.Conv2d(512, 512, 3, 1, 1, bias=True), nn.BatchNorm2d(512), nn.ReLU())
        self.max_pooling_4 = nn.MaxPool2d(kernel_size=2, stride=2, return_indices=True)

        # stage-5
        self.conv_5_1 = nn.Sequential(nn.Conv2d(256, 256, 3, 1, 1, bias=True), nn.BatchNorm2d(256), nn.ReLU())
        self.conv_5_2 = nn.Sequential(nn.Conv2d(256, 256, 3, 1, 1, bias=True), nn.BatchNorm2d(256), nn.ReLU())
        # self.conv_5_3 = nn.Sequential(nn.Conv2d(512, 512, 3, 1, 1, bias=True), nn.BatchNorm2d(512), nn.ReLU())

        # -----------------------------------------------------------------
        # alpha decoder
        # ---------------------
        # stage-5
        self.deconv_5 = nn.Sequential(nn.Conv2d(256, 256, 5, 1, 2, bias=True), nn.BatchNorm2d(256), nn.ReLU())

        # stage-4
        self.up_pool_4 = nn.MaxUnpool2d(2, stride=2)
        self.deconv_4 = nn.Sequential(nn.Conv2d(256, 128, 5, 1, 2, bias=True), nn.BatchNorm2d(128), nn.ReLU())

        # stage-3
        self.up_pool_3 = nn.MaxUnpool2d(2, stride=2)
        self.deconv_3 = nn.Sequential(nn.Conv2d(128, 64, 5, 1, 2, bias=True), nn.BatchNorm2d(64), nn.ReLU())

        # stage-2
        self.up_pool_2 = nn.MaxUnpool2d(2, stride=2)
        self.deconv_2 = nn.Sequential(nn.Conv2d(64, 32, 5, 1, 2, bias=True), nn.BatchNorm2d(32), nn.ReLU())

        # stage-1
        self.up_pool_1 = nn.MaxUnpool2d(2, stride=2)
        self.deconv_1 = nn.Sequential(nn.Conv2d(32, 32, 5, 1, 2, bias=True), nn.BatchNorm2d(32), nn.ReLU())

        # stage-0
        self.conv_0 = nn.Conv2d(32, 1, 5, 1, 2, bias=True)

        # alpha propagation
        self.bp1 = bottle(5, 128)
        self.p1 = BasicBlock(128, 128)

        self.bp2 = bottle(128, 256)
        self.p2 = BasicBlock(256, 256)

        self.a1 = nn.Conv2d(256, 1, 5, 1, 2, bias=True)

    def forward(self, x, trimap):
        alpha_input = torch.cat([x, trimap], dim=1)
        # ----------------
        # encoder
        # --------
        x11 = self.conv_1_1(alpha_input)
        x12 = self.conv_1_2(x11)
        x1p, id1 = self.max_pooling_1(x12)

        x21 = self.conv_2_1(x1p)
        x22 = self.conv_2_2(x21)
        x2p, id2 = self.max_pooling_2(x22)

        x31 = self.conv_3_1(x2p)
        x32 = self.conv_3_2(x31)
        # x33 = self.conv_3_3(x32)
        x3p, id3 = self.max_pooling_3(x32)

        x41 = self.conv_4_1(x3p)
        x42 = self.conv_4_2(x41)
        # x43 = self.conv_4_3(x42)
        x4p, id4 = self.max_pooling_4(x42)

        x51 = self.conv_5_1(x4p)
        x52 = self.conv_5_2(x51)
        # x53 = self.conv_5_3(x52)
        # ----------------
        # decoder
        # --------
        x5d = self.deconv_5(x52)

        x4u = self.up_pool_4(x5d, id4)
        x4d = self.deconv_4(x4u)

        x3u = self.up_pool_3(x4d, id3)
        x3d = self.deconv_3(x3u)

        x2u = self.up_pool_2(x3d, id2)
        x2d = self.deconv_2(x2u)

        x1u = self.up_pool_1(x2d, id1)
        x1d = self.deconv_1(x1u)

        # raw alpha pred
        a = self.conv_0(x1d)

        p1 = torch.cat([x, trimap, a], dim=1)
        # print('p1:', p1.size())
        p1 = self.bp1(p1)
        p1 = self.p1(p1)
        p1 = self.bp2(p1)
        p1 = self.p2(p1)
        a1 = self.a1(p1)

        p2 = torch.cat([x, trimap, a1], dim=1)
        p2 = self.bp1(p2)
        p2 = self.p1(p2)
        p2 = self.bp2(p2)
        p2 = self.p2(p2)
        a2 = self.a1(p2)

        p3 = torch.cat([x, trimap, a2], dim=1)
        p3 = self.bp1(p3)
        p3 = self.p1(p3)
        p3 = self.bp2(p3)
        p3 = self.p2(p3)
        a3 = self.a1(p3)

        return (a, a1, a2, a3)
