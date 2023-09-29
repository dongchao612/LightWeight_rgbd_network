#!/usr/bin/env python 
# -*- coding: utf-8 -*-
# Time    : 2023/9/29 13:19
# Author  : dongchao
# File    : LDN_net.py
# Software: PyCharm
import torch
import torch.nn as nn
import torch.nn.functional as F

from model.net_part import Swish, Hswish, SqueezeAndExciteFusionAdd, ConvBNAct
from model.resnet import ResNet18, ResNet34, ResNet50


class LDNNet(nn.Module):
    def __init__(self,
                 height=480,
                 width=640,
                 num_classes=37,
                 encoder_rgb='resnet18',
                 encoder_depth='resnet18',
                 encoder_block='BasicBlock',
                 channels_decoder=None,  # default: [128, 128, 128]
                 pretrained_on_imagenet=True,
                 pretrained_dir='./trained_models/imagenet',
                 activation='relu',
                 encoder_decoder_fusion='add',
                 fuse_depth_in_rgb_encoder='SE-add',
                 ):

        super(LDNNet, self).__init__()

        # set activation function
        if activation.lower() == 'relu':
            self.activation = nn.ReLU(inplace=True)
        elif activation.lower() in ['swish', 'silu']:
            self.activation = Swish()
        elif activation.lower() == 'hswish':
            self.activation = Hswish()
        else:
            raise NotImplementedError(
                'Only relu, swish and hswish as activation function are '
                'supported so far. Got {}'.format(activation))

        # rgb encoder
        if encoder_rgb == 'resnet18':
            self.encoder_rgb = ResNet18(
                block=encoder_block,
                pretrained_on_imagenet=pretrained_on_imagenet,
                pretrained_dir=pretrained_dir,
                activation=self.activation)
        elif encoder_rgb == 'resnet34':
            self.encoder_rgb = ResNet34(
                block=encoder_block,
                pretrained_on_imagenet=pretrained_on_imagenet,
                pretrained_dir=pretrained_dir,
                activation=self.activation)
        elif encoder_rgb == 'resnet50':
            self.encoder_rgb = ResNet50(
                pretrained_on_imagenet=pretrained_on_imagenet,
                activation=self.activation)

        # depth encoder
        if encoder_depth == 'resnet18':
            self.encoder_depth = ResNet18(
                block=encoder_block,
                pretrained_on_imagenet=pretrained_on_imagenet,
                pretrained_dir=pretrained_dir,
                activation=self.activation,
                input_channels=1)
        elif encoder_depth == 'resnet34':
            self.encoder_depth = ResNet34(
                block=encoder_block,
                pretrained_on_imagenet=pretrained_on_imagenet,
                pretrained_dir=pretrained_dir,
                activation=self.activation,
                input_channels=1)
        elif encoder_depth == 'resnet50':
            self.encoder_depth = ResNet50(
                pretrained_on_imagenet=pretrained_on_imagenet,
                activation=self.activation,
                input_channels=1)
        else:
            raise NotImplementedError(
                'Only ResNets are supported for '
                'encoder_depth. Got {}'.format(encoder_rgb))

        # rgb and depth add
        self.fuse_depth_in_rgb_encoder = fuse_depth_in_rgb_encoder
        if fuse_depth_in_rgb_encoder == 'SE-add':
            self.se_layer0 = SqueezeAndExciteFusionAdd(
                64, activation=self.activation)
            self.se_layer1 = SqueezeAndExciteFusionAdd(
                self.encoder_rgb.down_4_channels_out,
                activation=self.activation)
            self.se_layer2 = SqueezeAndExciteFusionAdd(
                self.encoder_rgb.down_8_channels_out,
                activation=self.activation)
            self.se_layer3 = SqueezeAndExciteFusionAdd(
                self.encoder_rgb.down_16_channels_out,
                activation=self.activation)
            self.se_layer4 = SqueezeAndExciteFusionAdd(
                self.encoder_rgb.down_32_channels_out,
                activation=self.activation)

        if channels_decoder is None:
            channels_decoder = [128, 128, 128]

        if encoder_decoder_fusion == 'add':
            layers_skip1 = list()
            if self.encoder_rgb.down_4_channels_out != channels_decoder[2]:
                layers_skip1.append(ConvBNAct(
                    self.encoder_rgb.down_4_channels_out,
                    channels_decoder[2],
                    kernel_size=1,
                    activation=self.activation))
            self.skip_layer1 = nn.Sequential(*layers_skip1)

            layers_skip2 = list()
            if self.encoder_rgb.down_8_channels_out != channels_decoder[1]:
                layers_skip2.append(ConvBNAct(
                    self.encoder_rgb.down_8_channels_out,
                    channels_decoder[1],
                    kernel_size=1,
                    activation=self.activation))
            self.skip_layer2 = nn.Sequential(*layers_skip2)

            layers_skip3 = list()
            if self.encoder_rgb.down_16_channels_out != channels_decoder[0]:
                layers_skip3.append(ConvBNAct(
                    self.encoder_rgb.down_16_channels_out,
                    channels_decoder[0],
                    kernel_size=1,
                    activation=self.activation))
            self.skip_layer3 = nn.Sequential(*layers_skip3)

    def forward(self, rgb, depth):
        rgb = self.encoder_rgb.forward_first_conv(rgb)
        depth = self.encoder_depth.forward_first_conv(depth)
        # print("rgb.shape,depth.shape", rgb.shape, depth.shape)
        if self.fuse_depth_in_rgb_encoder == 'add':
            fuse = rgb + depth
        else:
            fuse = self.se_layer0(rgb, depth)
        # print("fuse.shape", fuse.shape)
        rgb = F.max_pool2d(fuse, kernel_size=3, stride=2, padding=1)
        depth = F.max_pool2d(depth, kernel_size=3, stride=2, padding=1)

        # block 1
        rgb = self.encoder_rgb.forward_layer1(rgb)
        depth = self.encoder_depth.forward_layer1(depth)
        if self.fuse_depth_in_rgb_encoder == 'add':
            fuse = rgb + depth
        else:
            fuse = self.se_layer1(rgb, depth)
        skip1 = self.skip_layer1(fuse)
        # print("fuse.shape,skip1.shape", fuse.shape, skip1.shape)

        # block 2
        rgb = self.encoder_rgb.forward_layer2(fuse)
        depth = self.encoder_depth.forward_layer2(depth)
        if self.fuse_depth_in_rgb_encoder == 'add':
            fuse = rgb + depth
        else:
            fuse = self.se_layer2(rgb, depth)
        skip2 = self.skip_layer2(fuse)
        # print("fuse.shape, skip2.shape", fuse.shape, skip2.shape)

        # block 3
        rgb = self.encoder_rgb.forward_layer3(fuse)
        depth = self.encoder_depth.forward_layer3(depth)
        if self.fuse_depth_in_rgb_encoder == 'add':
            fuse = rgb + depth
        else:
            fuse = self.se_layer3(rgb, depth)
        skip3 = self.skip_layer3(fuse)
        # print("fuse.shape, skip3.shape", fuse.shape, skip3.shape)

        # block 4
        rgb = self.encoder_rgb.forward_layer4(fuse)
        depth = self.encoder_depth.forward_layer4(depth)
        if self.fuse_depth_in_rgb_encoder == 'add':
            fuse = rgb + depth
        else:
            fuse = self.se_layer4(rgb, depth)
        # print("fuse.shape", fuse.shape)

        # out = self.context_module(fuse)
        # out = self.decoder(enc_outs=[out, skip3, skip2, skip1])

        return fuse


if __name__ == '__main__':
    image_w = 640
    image_h = 480
    batch_size = 2

    model = LDNNet(
        height=image_h,
        width=image_w,
        fuse_depth_in_rgb_encoder="add",
        encoder_rgb='resnet34',
        encoder_depth='resnet34',
        encoder_block='NonBottleneck1D', pretrained_on_imagenet=True, pretrained_dir="../trained_models")

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(device)
    x = torch.randn(batch_size, 3, image_h, image_w)
    y = torch.randn(batch_size, 1, image_h, image_w)
    model.forward(x, y)
    from utils.utils import compute_speed_two
    compute_speed_two(model, (batch_size, 3, image_h, image_w), (batch_size, 1, image_h, image_w), device, 1)
    total = sum([param.nelement() for param in model.parameters()])
    print("Number of parameter: %.2fM" % (total / 1e6))
    '''
    =========Speed Testing=========
    Elapsed Time: [0.23 s / 10 iter]
    Speed Time: 23.14 ms / iter   FPS: 43.22
    Number of parameter: 42.6
    '''
