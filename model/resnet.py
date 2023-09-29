#!/usr/bin/env python 
# -*- coding: utf-8 -*-
# Time    : 2023/9/29 13:00
# Author  : dongchao
# File    : resnet.py
# Software: PyCharm
import os
from collections import OrderedDict
import pandas as pd
import torch
import torch.nn as nn

import torch.utils.model_zoo as model_zoo

from utils.utils import compute_speed_one, model_urls

import warnings

warnings.filterwarnings("ignore")



from model.net_part import conv1x1, BasicBlock, Bottleneck, NonBottleneck1D


class ResNet(nn.Module):

    def __init__(self, layers, block,
                 zero_init_residual=False, groups=1, width_per_group=64,
                 replace_stride_with_dilation=None, dilation=None,
                 norm_layer=None, input_channels=3,
                 activation=nn.ReLU(inplace=True)):
        super(ResNet, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer

        self.inplanes = 64
        self.dilation = 1
        if replace_stride_with_dilation is None:
            # each element in the tuple indicates if we should replace
            # the 2x2 stride with a dilated convolution instead
            replace_stride_with_dilation = [False, False, False]
        self.replace_stride_with_dilation = replace_stride_with_dilation
        if len(replace_stride_with_dilation) != 3:
            raise ValueError("replace_stride_with_dilation should be None "
                             "or a 3-element tuple, got "
                             "{}".format(replace_stride_with_dilation))
        if dilation is not None:
            if len(dilation) != 4:
                raise ValueError("dilation should be None "
                                 "or a 4-element tuple, got "
                                 "{}".format(dilation))
        else:
            dilation = [1, 1, 1, 1]

        self.groups = groups
        self.base_width = width_per_group
        self.conv1 = nn.Conv2d(input_channels, self.inplanes, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = norm_layer(self.inplanes)
        self.act = activation
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.down_2_channels_out = 64
        if self.replace_stride_with_dilation == [False, False, False]:
            self.down_4_channels_out = 64 * block.expansion
            self.down_8_channels_out = 128 * block.expansion
            self.down_16_channels_out = 256 * block.expansion
            self.down_32_channels_out = 512 * block.expansion
            # print(self.down_4_channels_out, self.down_8_channels_out, self.down_16_channels_out, self.down_32_channels_out)
            # 64 128 256 512
        elif self.replace_stride_with_dilation == [False, True, True]:
            self.down_4_channels_out = 64 * block.expansion
            self.down_8_channels_out = 512 * block.expansion

        # ************ make layer ************
        self.layer1 = self._make_layer(block, 64, layers[0], dilate=dilation[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2, dilate=dilation[1],
                                       replace_stride_with_dilation=replace_stride_with_dilation[0])
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2, dilate=dilation[2],
                                       replace_stride_with_dilation=replace_stride_with_dilation[1])
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2, dilate=dilation[3],
                                       replace_stride_with_dilation=replace_stride_with_dilation[2])
        # ************ init module ************
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight,
                                        mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        if zero_init_residual:
            for m in self.modules():
                nn.init.constant_(m.bn2.weight, 0)

    def _make_layer(self, block, planes, blocks,
                    stride=1, dilate=1, replace_stride_with_dilation=False):
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if replace_stride_with_dilation:
            self.dilation *= stride
            stride = 1
        if dilate > 1:
            self.dilation = dilate
            dilate_first_block = dilate
        else:
            dilate_first_block = previous_dilation
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                norm_layer(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample,
                            self.groups, self.base_width, dilate_first_block,
                            norm_layer,
                            activation=self.act))
        self.inplanes = planes * block.expansion

        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes,
                                groups=self.groups,
                                base_width=self.base_width,
                                dilation=self.dilation, norm_layer=norm_layer,
                                activation=self.act))

        return nn.Sequential(*layers)

    def forward(self, input):
        x = self.conv1(input)
        x = self.bn1(x)
        x_down2 = self.act(x)
        # print("x_down2 :",x_down2.shape)
        # torch.Size([3, 64, 240, 320])
        x = self.maxpool(x_down2)
        # print("after maxpool :",x.shape)
        # torch.Size([3, 64, 120, 160])

        x_layer1 = self.forward_resblock(x, self.layer1)
        x_layer2 = self.forward_resblock(x_layer1, self.layer2)
        x_layer3 = self.forward_resblock(x_layer2, self.layer3)
        x_layer4 = self.forward_resblock(x_layer3, self.layer4)
        # print(x_layer1.shape, x_layer2.shape, x_layer3.shape, x_layer4.shape)
        # torch.Size([3, 64, 120, 160]) torch.Size([3, 128, 60, 80]) torch.Size([3, 256, 30, 40]) torch.Size([3, 512, 15, 20])
        features = []
        if self.replace_stride_with_dilation == [False, False, False]:
            features = [x_layer4, x_layer3, x_layer2, x_layer1]
            # print(x_layer1.size(),x_layer2.size(),x_layer3.size(),x_layer4.size())
            self.skip3_channels = x_layer3.size()[1]
            self.skip2_channels = x_layer2.size()[1]
            self.skip1_channels = x_layer1.size()[1]
            # print(self.skip1_channels, self.skip2_channels, self.skip3_channels)
            # 64 128 256

        elif self.replace_stride_with_dilation == [False, True, True]:
            # x has resolution 1/8
            # skip4 has resolution 1/8
            # skip3 has resolution 1/8
            # skip2 has resolution 1/8
            # skip1 has resolution 1/4
            # x_down2 has resolution 1/2
            features = [x, x_layer1, x_down2]

            self.skip3_channels = x_layer3.size()[1]
            self.skip2_channels = x_layer2.size()[1]
            self.skip1_channels = x_layer1.size()[1]
            # print(self.skip1_channels, self.skip2_channels, self.skip3_channels)

        return features

    def forward_resblock(self, x, layers):
        for l in layers:
            x = l(x)
        return x

    def forward_first_conv(self, x):
        # be aware that maxpool still needs to be applied after this function
        # and before forward_layer1()
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.act(x)
        return x

    def forward_layer1(self, x):
        # be ware that maxpool still needs to be applied after
        # forward_first_conv() and before this function
        x = self.forward_resblock(x, self.layer1)
        self.skip1_channels = x.size()[1]
        return x

    def forward_layer2(self, x):
        x = self.forward_resblock(x, self.layer2)
        self.skip2_channels = x.size()[1]
        return x

    def forward_layer3(self, x):
        x = self.forward_resblock(x, self.layer3)
        self.skip3_channels = x.size()[1]
        return x

    def forward_layer4(self, x):
        x = self.forward_resblock(x, self.layer4)
        return x


def ResNet18(pretrained_on_imagenet=False,
             pretrained_dir='./trained_models',
             **kwargs):
    if 'block' not in kwargs:
        kwargs['block'] = BasicBlock
    else:
        if kwargs['block'] in globals():
            # convert string to block object
            kwargs['block'] = globals()[kwargs['block']]

        else:
            raise NotImplementedError('Block {} is not implemented'
                                      ''.format(kwargs['block']))
    model = ResNet([2, 2, 2, 2], **kwargs)
    if 'input_channels' in kwargs and kwargs['input_channels'] == 1:
        input_channels = 1
    else:
        input_channels = 3
    if kwargs['block'] != BasicBlock and pretrained_on_imagenet:
        model = load_pretrained_with_different_encoder_block(
            model, kwargs['block'].__name__,
            input_channels, 'r18',
            pretrained_dir=pretrained_dir
        )
    elif pretrained_on_imagenet:
        weights = model_zoo.load_url(model_urls['resnet18'], model_dir='./')
        if 'input_channels' in kwargs and kwargs['input_channels'] == 1:
            # sum the weights of the first convolution
            weights['conv1.weight'] = torch.sum(weights['conv1.weight'],
                                                dim=1, keepdim=True)
        weights.pop('fc.weight')
        weights.pop('fc.bias')
        model.load_state_dict(weights, strict=True)
        print('Loaded ResNet18 pretrained on ImageNet')
    return model


def ResNet34(pretrained_on_imagenet=False,
             pretrained_dir='./trained_models',
             **kwargs):
    if 'block' not in kwargs:
        kwargs['block'] = BasicBlock
    else:
        if kwargs['block'] in globals():
            # convert string to block object
            kwargs['block'] = globals()[kwargs['block']]

        else:
            raise NotImplementedError('Block {} is not implemented'
                                      ''.format(kwargs['block']))
    if 'input_channels' in kwargs and kwargs['input_channels'] == 1:
        input_channels = 1
    else:
        input_channels = 3
    model = ResNet([3, 4, 6, 3], **kwargs)

    if kwargs['block'] != BasicBlock and pretrained_on_imagenet:
        model = load_pretrained_with_different_encoder_block(
            model, kwargs['block'].__name__,
            input_channels, 'r34',
            pretrained_dir=pretrained_dir
        )
    elif pretrained_on_imagenet:
        weights = model_zoo.load_url(model_urls['resnet34'], model_dir='./')
        if input_channels == 1:
            # sum the weights of the first convolution
            weights['conv1.weight'] = torch.sum(weights['conv1.weight'],
                                                dim=1, keepdim=True)
        weights.pop('fc.weight')
        weights.pop('fc.bias')
        model.load_state_dict(weights, strict=True)
        print('Loaded ResNet34 pretrained on ImageNet')
    return model


def ResNet50(pretrained_on_imagenet=False, **kwargs):
    if 'block' not in kwargs:
        kwargs['block'] = BasicBlock
    else:
        if kwargs['block'] in globals():
            # convert string to block object
            kwargs['block'] = globals()[kwargs['block']]

        else:
            raise NotImplementedError('Block {} is not implemented'
                                      ''.format(kwargs['block']))
    if 'input_channels' in kwargs and kwargs['input_channels'] == 1:
        input_channels = 1
    else:
        input_channels = 3

    model = ResNet([3, 4, 6, 3], **kwargs)

    if pretrained_on_imagenet:
        weights = model_zoo.load_url(model_urls['resnet50'], model_dir='./')
        if input_channels == 1:
            # sum the weights of the first convolution
            weights['conv1.weight'] = torch.sum(weights['conv1.weight'],
                                                dim=1, keepdim=True)
        weights.pop('fc.weight')
        weights.pop('fc.bias')
        model.load_state_dict(weights, strict=True)
        print('Loaded ResNet50 pretrained on ImageNet')
    return model


def load_pretrained_with_different_encoder_block(
        model, encoder_block, input_channels, resnet_name,
        pretrained_dir='./trained_models'):
    ckpt_path = os.path.join(pretrained_dir, '{}_NBt1D.pth'.format(resnet_name))

    if not os.path.exists(ckpt_path):
        # get best weights file from logs
        logs = pd.read_csv(os.path.join(pretrained_dir, 'logs.csv'))
        idx_top1 = logs['acc_val_top-1'].idxmax()
        acc_top1 = logs['acc_val_top-1'][idx_top1]
        epoch = logs.epoch[idx_top1]
        ckpt_path = os.path.join(pretrained_dir,
                                 'ckpt_epoch_{}.pth'.format(epoch))
        print("Choosing checkpoint {} with top1 acc {}".format(ckpt_path, acc_top1))

    # load weights
    if torch.cuda.is_available():
        checkpoint = torch.load(ckpt_path)
    else:
        checkpoint = torch.load(ckpt_path, map_location=torch.device('cpu'))
    checkpoint['state_dict2'] = OrderedDict()

    # rename keys and leave out last fully connected layer
    for key in checkpoint['state_dict']:
        if 'encoder' in key:
            checkpoint['state_dict2'][key.split('encoder.')[-1]] = \
                checkpoint['state_dict'][key]
    weights = checkpoint['state_dict2']

    if input_channels == 1:
        # sum the weights of the first convolution
        weights['encoder_depth.conv1.weight'] = torch.sum(weights['encoder_depth.conv1.weight'],
                                            dim=1,
                                            keepdim=True)

    model.load_state_dict(weights, strict=False)
    print('Loaded {} with encoder block   {}pretrained on ImageNet'.format(resnet_name, encoder_block))
    print(ckpt_path)
    return model


if __name__ == '__main__':
    image_h, image_w = 480, 640
    batch_size = 3
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    # print(device)
    #
    # print("r18 ")
    # model18 = ResNet18(block='BasicBlock', pretrained_on_imagenet=True,
    #                   dilation=[1]*4)
    # compute_speed_one(model18, (batch_size, 3, image_h, image_w), device, 10)
    # total = sum([param.nelement() for param in model18.parameters()])
    # print("Number of parameter: %.2fM" % (total / 1e6))
    #
    # model18 = ResNet18(block='Bottleneck', pretrained_on_imagenet=True,
    #                   dilation=[1]*4)
    # compute_speed_one(model18, (batch_size, 3, image_h, image_w), device, 10)
    # total = sum([param.nelement() for param in model18.parameters()])
    # print("Number of parameter: %.2fM" % (total / 1e6))
    #
    # model18 = ResNet18(block='NonBottleneck1D', pretrained_on_imagenet=True,
    #                   dilation=[1]*4)
    # compute_speed_one(model18, (batch_size, 3, image_h, image_w), device, 10)
    # total = sum([param.nelement() for param in model18.parameters()])
    # print("Number of parameter: %.2fM" % (total / 1e6))

    # print("r34 ")
    ##  Bottleneck NonBottleneck1Dto load r34_NBt1D
    ##  BasicBlock to load resnet34-333f7ec4

    # model34 = ResNet34(block='BasicBlock', pretrained_on_imagenet=True,
    #                    dilation=[1] * 4, pretrained_dir="../trained_models")
    # compute_speed_one(model34, (batch_size, 3, image_h, image_w), device, 10)
    # total = sum([param.nelement() for param in model34.parameters()])
    # print("BasicBlock Number of parameter: %.2fM" % (total / 1e6))
    #
    # model34 = ResNet34(block='Bottleneck', pretrained_on_imagenet=True,
    #                    dilation=[1] * 4, pretrained_dir="../trained_models")
    # compute_speed_one(model34, (batch_size, 3, image_h, image_w), device, 10)
    # total = sum([param.nelement() for param in model34.parameters()])
    # print("Bottleneck Number of parameter: %.2fM" % (total / 1e6))
    #
    model34 = ResNet34(block='NonBottleneck1D', pretrained_on_imagenet=True,
                       dilation=[1] * 4, pretrained_dir="../trained_models")
    compute_speed_one(model34, (batch_size, 3, image_h, image_w), device, 10)
    total = sum([param.nelement() for param in model34.parameters()])
    print("NonBottleneck1D Number of parameter: %.2fM" % (total / 1e6))
    #
    # print("r50 ")
    # model50 = ResNet50(block='BasicBlock', pretrained_on_imagenet=False,
    #                   dilation=[1]*4)
    # compute_speed_one(model50, (batch_size, 3, image_h, image_w), device, 10)
    # total = sum([param.nelement() for param in model50.parameters()])
    # print("Number of parameter: %.2fM" % (total / 1e6))
    #
    # model50 = ResNet50(block='Bottleneck', pretrained_on_imagenet=False,
    #                   dilation=[1]*4)
    # compute_speed_one(model50, (batch_size, 3, image_h, image_w), device, 10)
    # total = sum([param.nelement() for param in model50.parameters()])
    # print("Number of parameter: %.2fM" % (total / 1e6))

    # model50 = ResNet50(block='NonBottleneck1D', pretrained_on_imagenet=False,
    #                   dilation=[1]*4)
    # compute_speed_one(model50, (batch_size, 3, image_h, image_w), device, 10)
    # total = sum([param.nelement() for param in model50.parameters()])
    # print("Number of parameter: %.2fM" % (total / 1e6))

    # BasicBlock+True   resnet34-333f7ec4
    model34 = ResNet34(block='NonBottleneck1D', pretrained_on_imagenet=True,
                       dilation=[1] * 4, pretrained_dir="../trained_models")
    x = torch.randn(batch_size, 3, image_h, image_w)
    for i in model34(x):
        print(i.shape)
    '''
    torch.Size([3, 512, 15, 20])
    torch.Size([3, 256, 30, 40])
    torch.Size([3, 128, 60, 80])
    torch.Size([3, 64, 120, 160])
    '''
