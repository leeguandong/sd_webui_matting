import torch
import torch.nn as nn
from torchvision import models
import torch.nn.functional as F
import warnings
# from mmcv.cnn import ConvModule

from .swin_transformer import swinB, swinS, swinT


class RefUnet(nn.Module):
    def __init__(self,in_ch,inc_ch):
        super(RefUnet, self).__init__()

        self.conv0 = nn.Conv2d(in_ch,inc_ch,3,padding=1)

        self.conv1 = nn.Conv2d(inc_ch,64,3,padding=1)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu1 = nn.ReLU(inplace=True)

        self.pool1 = nn.MaxPool2d(2,2,ceil_mode=True)

        self.conv2 = nn.Conv2d(64,64,3,padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.relu2 = nn.ReLU(inplace=True)

        self.pool2 = nn.MaxPool2d(2,2,ceil_mode=True)

        self.conv3 = nn.Conv2d(64,64,3,padding=1)
        self.bn3 = nn.BatchNorm2d(64)
        self.relu3 = nn.ReLU(inplace=True)

        self.pool3 = nn.MaxPool2d(2,2,ceil_mode=True)

        self.conv4 = nn.Conv2d(64,64,3,padding=1)
        self.bn4 = nn.BatchNorm2d(64)
        self.relu4 = nn.ReLU(inplace=True)

        self.pool4 = nn.MaxPool2d(2,2,ceil_mode=True)

        #####

        self.conv5 = nn.Conv2d(64,64,3,padding=1)
        self.bn5 = nn.BatchNorm2d(64)
        self.relu5 = nn.ReLU(inplace=True)

        #####

        self.conv_d4 = nn.Conv2d(128,64,3,padding=1)
        self.bn_d4 = nn.BatchNorm2d(64)
        self.relu_d4 = nn.ReLU(inplace=True)

        self.conv_d3 = nn.Conv2d(128,64,3,padding=1)
        self.bn_d3 = nn.BatchNorm2d(64)
        self.relu_d3 = nn.ReLU(inplace=True)

        self.conv_d2 = nn.Conv2d(128,64,3,padding=1)
        self.bn_d2 = nn.BatchNorm2d(64)
        self.relu_d2 = nn.ReLU(inplace=True)

        self.conv_d1 = nn.Conv2d(128,64,3,padding=1)
        self.bn_d1 = nn.BatchNorm2d(64)
        self.relu_d1 = nn.ReLU(inplace=True)

        self.conv_d0 = nn.Conv2d(64,1,3,padding=1)

        self.upscore2 = nn.Upsample(scale_factor=2, mode='bilinear')

    def forward(self,x):

        hx = x
        hx = self.conv0(hx)

        hx1 = self.relu1(self.bn1(self.conv1(hx)))
        hx = self.pool1(hx1)

        hx2 = self.relu2(self.bn2(self.conv2(hx)))
        hx = self.pool2(hx2)

        hx3 = self.relu3(self.bn3(self.conv3(hx)))
        hx = self.pool3(hx3)

        hx4 = self.relu4(self.bn4(self.conv4(hx)))
        hx = self.pool4(hx4)

        hx5 = self.relu5(self.bn5(self.conv5(hx)))

        hx = self.upscore2(hx5)

        d4 = self.relu_d4(self.bn_d4(self.conv_d4(torch.cat((hx,hx4),1))))
        hx = self.upscore2(d4)

        d3 = self.relu_d3(self.bn_d3(self.conv_d3(torch.cat((hx,hx3),1))))
        hx = self.upscore2(d3)

        d2 = self.relu_d2(self.bn_d2(self.conv_d2(torch.cat((hx,hx2),1))))
        hx = self.upscore2(d2)

        d1 = self.relu_d1(self.bn_d1(self.conv_d1(torch.cat((hx,hx1),1))))

        residual = self.conv_d0(d1)

        return x + residual

def resize(input,
           size=None,
           scale_factor=None,
           mode='nearest',
           align_corners=None,
           warning=True):
    if warning:
        if size is not None and align_corners:
            input_h, input_w = tuple(int(x) for x in input.shape[2:])
            output_h, output_w = tuple(int(x) for x in size)
            if output_h > input_h or output_w > output_h:
                if ((output_h > 1 and output_w > 1 and input_h > 1
                     and input_w > 1) and (output_h - 1) % (input_h - 1)
                        and (output_w - 1) % (input_w - 1)):
                    warnings.warn(
                        f'When align_corners={align_corners}, '
                        'the output would more aligned if '
                        f'input size {(input_h, input_w)} is `x+1` and '
                        f'out size {(output_h, output_w)} is `nx+1`')
    if isinstance(size, torch.Size):
        size = tuple(int(x) for x in size)
    return F.interpolate(input, size, scale_factor, mode, align_corners)

class PPM(nn.ModuleList):
    """Pooling Pyramid Module used in PSPNet.

    Args:
        pool_scales (tuple[int]): Pooling scales used in Pooling Pyramid
            Module.
        in_channels (int): Input channels.
        channels (int): Channels after modules, before conv_seg.
        conv_cfg (dict|None): Config of conv layers.
        norm_cfg (dict|None): Config of norm layers.
        act_cfg (dict): Config of activation layers.
        align_corners (bool): align_corners argument of F.interpolate.
    """

    def __init__(self, pool_scales, in_channels, channels=512, conv_cfg=None, norm_cfg=None,
                 act_cfg=dict(type='ReLU'), align_corners=False):
        super(PPM, self).__init__()
        self.pool_scales = pool_scales
        self.align_corners = align_corners
        self.in_channels = in_channels
        self.channels = channels
        self.conv_cfg = conv_cfg
        self.norm_cfg = norm_cfg
        self.act_cfg = act_cfg
        # self.pool_layers = []
        for pool_scale in pool_scales:
            # self.pool_layers.append(
            self.append(
                nn.Sequential(
                    nn.AdaptiveAvgPool2d(pool_scale),
                    ConvModule(
                        self.in_channels,
                        self.channels,
                        1,
                        conv_cfg=self.conv_cfg,
                        norm_cfg=self.norm_cfg,
                        act_cfg=self.act_cfg)))
        # self.bottleneck = ConvModule(
        #     self.in_channels + 4 * self.channels,
        #     self.channels,
        #     3,
        #     padding=1,
        #     conv_cfg=self.conv_cfg,
        #     norm_cfg=self.norm_cfg,
        #     act_cfg=self.act_cfg)

    def forward(self, x):
        """Forward function."""
        # ppm_outs = [x]
        # for ppm in self.pool_layers:
        ppm_outs = []
        for ppm in self:
            ppm_out = ppm(x)
            upsampled_ppm_out = resize(
                ppm_out,
                size=x.size()[2:],
                mode='bilinear',
                align_corners=self.align_corners)  # 恢复为输入大小，以便后续做concat
            ppm_outs.append(upsampled_ppm_out)
        # ppm_outs = torch.cat(ppm_outs, dim=1)
        # ppm_outs = self.bottleneck(ppm_outs)
        return ppm_outs



class SwinBASNet(nn.Module):
    def __init__(self, channels=512, n_classes=1, dropout_ratio=0.1, pretrained=None):
        super(SwinBASNet, self).__init__()

        ## -------------Encoder--------------
        swin = swinB(pretrained=pretrained)
        self.encoder = swin

        self.channels = channels
        self.in_channels = [self.encoder.embed_dim, self.encoder.embed_dim*2, self.encoder.embed_dim*4, self.encoder.embed_dim*8]

        ## ----------PPM (pyramid pooling module)------#
        self.psp_modules = PPM(
            pool_scales=(1, 2, 3, 6),
            in_channels=self.in_channels[-1],
            channels=self.channels)        # pool_size = [1*1 2*2 3*3 6*6]  outsize batch_size * 512 * 原始尺寸
        self.bottleneck = ConvModule(
            self.in_channels[-1] + 4 * self.channels,
            self.channels,
            3,
            padding=1)
        
        ## --------------FPN-----------------#
        self.lateral_convs = nn.ModuleList()
        self.fpn_convs = nn.ModuleList()
        for in_channels in self.in_channels[:-1]:  # skip the top layer
            l_conv = ConvModule(
                in_channels,
                self.channels,
                1,
                inplace=False)
            fpn_conv = ConvModule(
                self.channels,
                self.channels,
                3,
                padding=1,
                inplace=False)
            self.lateral_convs.append(l_conv)
            self.fpn_convs.append(fpn_conv)

        self.fpn_bottleneck = ConvModule(
            len(self.in_channels) * self.channels,
            self.channels,
            3,
            padding=1)
        
        ## ------------ seg cls head -----------------#
        self.conv_seg = nn.Conv2d(self.channels, n_classes, kernel_size=1)
        if dropout_ratio > 0:
            self.dropout = nn.Dropout2d(dropout_ratio)
        else:
            self.dropout = None
        
        ## -------------Side Output--------------
        self.aux_bottleneck_0 = ConvModule(
            self.channels,
            n_classes,
            3,
            padding=1)
        
        self.aux_bottleneck_1 = ConvModule(
            self.channels,
            n_classes,
            3,
            padding=1)

        self.aux_bottleneck_2 = ConvModule(
            self.channels,
            n_classes,
            3,
            padding=1)
        self.aux_bottleneck_3 = ConvModule(
            self.channels,
            n_classes,
            3,
            padding=1)

    ## -------------Bilinear Upsampling--------------
        self.upscore = nn.Upsample(scale_factor=4, mode='bilinear')
    
    ## -------------Refine Module-------------
        self.refunet = RefUnet(1,64)
    
    def psp_forward(self, inputs):
        """Forward function of PSP module."""
        x = inputs[-1]
        psp_outs = [x]
        psp_outs.extend(self.psp_modules(x))
        psp_outs = torch.cat(psp_outs, dim=1)
        output = self.bottleneck(psp_outs)
        return output
    
    def cls_seg(self, feat):
        """Classify each pixel."""
        if self.dropout is not None:
            feat = self.dropout(feat)
        output = self.conv_seg(feat)
        return output

    def forward(self,x):
        ## -------------Encoder-------------
        inputs = self.encoder(x) # x4 x8 x16 x32  (对于448的输入，输出为112 56 28 14)

        # build laterals
        laterals = [
            lateral_conv(inputs[i])
            for i, lateral_conv in enumerate(self.lateral_convs)
        ]   # 统一512个通道

        laterals.append(self.psp_forward(inputs))  # 加入顶层PPM的输出

        # build top-down path
        used_backbone_levels = len(laterals)
        for i in range(used_backbone_levels - 1, 0, -1):
            prev_shape = laterals[i - 1].shape[2:]
            laterals[i - 1] += resize(
                laterals[i],
                size=prev_shape,
                mode='bilinear')

        # build outputs
        fpn_outs = [
            self.fpn_convs[i](laterals[i])
            for i in range(used_backbone_levels - 1)
        ]

        # append psp feature
        fpn_outs.append(laterals[-1])

        for i in range(used_backbone_levels - 1, 0, -1):
            fpn_outs[i] = resize(
                fpn_outs[i],
                size=fpn_outs[0].shape[2:],
                mode='bilinear')        # 所有的特征图resize到原来的1/4

        # for i in range(used_backbone_levels):
        #     fpn_outs[i] = resize(
        #         fpn_outs[i],
        #         size=fpn_outs[0].shape[2:],
        #         mode='bilinear')        # 所有的特征图resize到原来的1/4

        # d1 = self.upscore(self.aux_bottleneck_0(fpn_outs[0]))  # 上采样到原图大小
        # d2 = self.upscore(self.aux_bottleneck_1(fpn_outs[1]))  # 上采样到原图大小
        # d3 = self.upscore(self.aux_bottleneck_2(fpn_outs[2]))  # 上采样到原图大小
        # d4 = self.upscore(self.aux_bottleneck_3(fpn_outs[3]))  # 上采样到原图大小

        fpn_outs = torch.cat(fpn_outs, dim=1)
        output = self.fpn_bottleneck(fpn_outs)
        output = self.cls_seg(output)

        ## -------------Refine Module-------------
        # d0 = self.upscore(output)   # 上采样到原图大小
        # dout = self.refunet(d0) # 256

        dout = self.upscore(output)   # 上采样到原图大小

        return F.sigmoid(dout)#, F.sigmoid(d1), F.sigmoid(d2), F.sigmoid(d3), F.sigmoid(d4)


if __name__ == '__main__':
    import numpy as np
    import time
    swin_b = SwinBASNet(n_classes=1)
    swin_b.cuda()
    # print(swin_b)
    img = torch.Tensor(np.random.random((1, 3, 640, 640)))
    print(img.shape, img.dtype)
    swin_b.eval()
    st_time = time.time()
    with torch.no_grad():
        out = swin_b(img.cuda())
        print(out.shape)
        # print(out[0].shape)
        # print(out[1].shape)
        # print(out[2].shape)
        # print(out[3].shape)
        # print(out[4].shape)
    print("cost_time: %fs"%(time.time()-st_time))