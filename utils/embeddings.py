# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Parameter as P
from torchvision.models.inception import inception_v3
from torchvision.models import vgg16


class IdentityEmbedding:
    def __call__(self, y):
        return y


class OneHotEmbedding:
    def __init__(self, num_classes):
        self.num_classes = num_classes

    def __call__(self, y):
        # Expects to receive a 1D array of numbers
        onehot = torch.eye(self.num_classes)[y].cuda()
        return onehot


class InceptionEmbedding:
    def __init__(self, parallel=False):
        # Expects inputs to be in range [-1, 1]
        inception_model = inception_v3(pretrained=True, transform_input=False)
        inception_model = WrapInception(inception_model.eval()).cuda()
        if parallel:
            inception_model = nn.DataParallel(inception_model)
        self.inception_model = inception_model

    def __call__(self, x):
        return self.inception_model(x)


# Wrapper for Inceptionv3, from Andrew Brock (modified slightly)
# https://github.com/ajbrock/BigGAN-PyTorch/blob/master/inception_utils.py
class WrapInception(nn.Module):
    def __init__(self, net):
        super(WrapInception, self).__init__()
        self.net = net
        self.mean = P(torch.tensor([0.485, 0.456, 0.406]).view(1, -1, 1, 1),
                      requires_grad=False)
        self.std = P(torch.tensor([0.229, 0.224, 0.225]).view(1, -1, 1, 1),
                     requires_grad=False)

    def forward(self, x):
        # Normalize x
        x = (x + 1.) / 2.0  # assume the input is normalized to [-1, 1], reset it to [0, 1]
        x = (x - self.mean) / self.std
        # Upsample if necessary
        if x.shape[2] != 299 or x.shape[3] != 299:
            x = F.interpolate(x, size=(299, 299), mode='bilinear', align_corners=True)
        # 299 x 299 x 3
        x = self.net.Conv2d_1a_3x3(x)
        # 149 x 149 x 32
        x = self.net.Conv2d_2a_3x3(x)
        # 147 x 147 x 32
        x = self.net.Conv2d_2b_3x3(x)
        # 147 x 147 x 64
        x = F.max_pool2d(x, kernel_size=3, stride=2)
        # 73 x 73 x 64
        x = self.net.Conv2d_3b_1x1(x)
        # 73 x 73 x 80
        x = self.net.Conv2d_4a_3x3(x)
        # 71 x 71 x 192
        x = F.max_pool2d(x, kernel_size=3, stride=2)
        # 35 x 35 x 192
        x = self.net.Mixed_5b(x)
        # 35 x 35 x 256
        x = self.net.Mixed_5c(x)
        # 35 x 35 x 288
        x = self.net.Mixed_5d(x)
        # 35 x 35 x 288
        x = self.net.Mixed_6a(x)
        # 17 x 17 x 768
        x = self.net.Mixed_6b(x)
        # 17 x 17 x 768
        x = self.net.Mixed_6c(x)
        # 17 x 17 x 768
        x = self.net.Mixed_6d(x)
        # 17 x 17 x 768
        x = self.net.Mixed_6e(x)
        # 17 x 17 x 768
        # 17 x 17 x 768
        x = self.net.Mixed_7a(x)
        # 8 x 8 x 1280
        x = self.net.Mixed_7b(x)
        # 8 x 8 x 2048
        x = self.net.Mixed_7c(x)
        # 8 x 8 x 2048
        pool = F.adaptive_avg_pool2d(x, 1).view(x.size(0), -1)
        # 1 x 1 x 2048
        return pool

class VGG16Embedding:
    def __init__(self, parallel=False):
        # Expects inputs to be in range [-1, 1]
        vgg_model = vgg16(pretrained=True).eval()
        vgg_model = WrapVGG(vgg_model).cuda()
        if parallel:
            vgg_model = nn.DataParallel(vgg_model)

        self.vgg_model = vgg_model

    def __call__(self, x):
        return self.vgg_model(x)

class WrapVGG(nn.Module):
    def __init__(self, net):
        super(WrapVGG, self).__init__()
        self.features = list(net.features)
        self.features = nn.Sequential(*self.features)
        # Extract VGG-16 Average Pooling Layer
        # self.pooling = net.avgpool
        self.pooling = nn.AdaptiveAvgPool2d((1, 1))

        # Convert the image into one-dimensional vector
        self.flatten = nn.Flatten()
        # Extract the first part of fully-connected layer from VGG16
        self.fc = net.classifier[:-5]
        print(self.fc)

        # net.classifier = net.classifier[:-1]
        self.net = net
        self.mean = P(torch.tensor([0.485, 0.456, 0.406]).view(1, -1, 1, 1),
                      requires_grad=False)
        self.std = P(torch.tensor([0.229, 0.224, 0.225]).view(1, -1, 1, 1),
                     requires_grad=False)

    def forward(self, x):
        # Normalize x
        # x = (x + 1.) / 2.0  # assume the input is normalized to [-1, 1], reset it to [0, 1]

        # if x.shape[2] != 256 or x.shape[3] != 256:
        #     x = F.interpolate(x, size=(256, 256), mode='bilinear', align_corners=True)
            # x = TF.resize(x, 256)
        #
        # x = TF.center_crop(x, 224)
        #
        # x = (x - self.mean) / self.std

        # Upsample if necessary
        # if x.shape[2] != 224 or x.shape[3] != 224:
        #     x = F.interpolate(x, size=(224, 224), mode='bilinear', align_corners=True)

        out = self.features(x)
        # out = self.pooling(out)
        out = self.pooling(out).view(x.size(0), -1)
        # out = self.flatten(out)
        # out = self.fc(out)
        return out
