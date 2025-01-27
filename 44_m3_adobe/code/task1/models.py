# -*- coding: utf-8 -*-
"""Implements SRGAN models: https://arxiv.org/abs/1609.04802

TODO:

"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable


def swish(x):
    return x * F.sigmoid(x)

class FeatureExtractor(nn.Module):
    def __init__(self, cnn, feature_layer=11):
        super(FeatureExtractor, self).__init__()
        self.features = nn.Sequential(*list(cnn.features.children())[:(feature_layer+1)])

    def forward(self, x):
        return self.features(x)


class residualBlock(nn.Module):
    def __init__(self, in_channels=64, k=3, n=64, s=1):
        super(residualBlock, self).__init__()

        self.conv1 = nn.Conv2d(in_channels, n, k, stride=s, padding=1)
        self.bn1 = nn.BatchNorm2d(n)
        self.conv2 = nn.Conv2d(n, n, k, stride=s, padding=1)
        self.bn2 = nn.BatchNorm2d(n)

    def forward(self, x):
        y = swish(self.bn1(self.conv1(x)))
        return self.bn2(self.conv2(y)) + x

class upsampleBlock(nn.Module):
    # Implements resize-convolution
    def __init__(self, in_channels, out_channels):
        super(upsampleBlock, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, 3, stride=1, padding=1)
        self.shuffler = nn.PixelShuffle(2)

    def forward(self, x):
        return swish(self.shuffler(self.conv(x)))

class Generator(nn.Module):
    def __init__(self, n_residual_blocks, upsample_factor):
        super(Generator, self).__init__()
        self.n_residual_blocks = n_residual_blocks
        self.upsample_factor = upsample_factor

        self.conv1 = nn.Conv2d(3, 64, 9, stride=1, padding=4)

        for i in range(self.n_residual_blocks):
            self.add_module('residual_block' + str(i+1), residualBlock())

        self.conv2 = nn.Conv2d(64, 64, 3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(64)

        for i in range(self.upsample_factor//2):
            self.add_module('upsample' + str(i+1), upsampleBlock(64, 256))

        self.conv3 = nn.Conv2d(64, 3, 9, stride=1, padding=4)

    def forward(self, x):
        x = swish(self.conv1(x))

        y = x.clone()
        for i in range(self.n_residual_blocks):
            y = self.__getattr__('residual_block' + str(i+1))(y)

        x = self.bn2(self.conv2(y)) + x

        for i in range(self.upsample_factor//2):
            x = self.__getattr__('upsample' + str(i+1))(x)

        return self.conv3(x)

class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, 3, stride=1, padding=1)

        self.conv2 = nn.Conv2d(64, 64, 3, stride=2, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 128, 3, stride=1, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        self.conv4 = nn.Conv2d(128, 128, 3, stride=2, padding=1)
        self.bn4 = nn.BatchNorm2d(128)
        self.conv5 = nn.Conv2d(128, 256, 3, stride=1, padding=1)
        self.bn5 = nn.BatchNorm2d(256)
        self.conv6 = nn.Conv2d(256, 256, 3, stride=2, padding=1)
        self.bn6 = nn.BatchNorm2d(256)
        self.conv7 = nn.Conv2d(256, 512, 3, stride=1, padding=1)
        self.bn7 = nn.BatchNorm2d(512)
        self.conv8 = nn.Conv2d(512, 512, 3, stride=2, padding=1)
        self.bn8 = nn.BatchNorm2d(512)

        # Replaced original paper FC layers with FCN
        self.conv9 = nn.Conv2d(512, 1, 1, stride=1, padding=1)

    def forward(self, x):
        x = swish(self.conv1(x))

        x = swish(self.bn2(self.conv2(x)))
        x = swish(self.bn3(self.conv3(x)))
        x = swish(self.bn4(self.conv4(x)))
        x = swish(self.bn5(self.conv5(x)))
        x = swish(self.bn6(self.conv6(x)))
        x = swish(self.bn7(self.conv7(x)))
        x = swish(self.bn8(self.conv8(x)))

        x = self.conv9(x)
        return F.sigmoid(F.avg_pool2d(x, x.size()[2:])).view(x.size()[0], -1)



class AutoEncoderConcat(nn.Module):
    def __init__(self):
        super().__init__()
        
        # Encoder 
        self.encoder1 = nn.Conv2d(3, 32, kernel_size=3, stride=2, padding=1)
        self.relu1 = nn.ReLU()
        self.encoder2 = nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1)
        self.relu2 = nn.ReLU()
        self.encoder3 = nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1)
        self.relu3 = nn.ReLU()
        self.encoder4 = nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1)
        self.relu4 = nn.ReLU()
        self.encoder5 = nn.Conv2d(256, 512, kernel_size=3, stride=2, padding=1)
        self.relu5 = nn.ReLU()
        self.drop1 = nn.Dropout(0.7)
        self.drop2 = nn.Dropout(0.7)
        self.drop3 = nn.Dropout(0.7)
        self.drop4 = nn.Dropout(0.7)
        
        # Decoder
        self.decoder1 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2, padding=0)
        self.derelu1 = nn.ReLU()
        self.decoder2 = nn.ConvTranspose2d(512, 128, kernel_size=2, stride=2, padding=0)
        self.derelu2 = nn.ReLU()
        self.decoder3 = nn.ConvTranspose2d(256, 64, kernel_size=2, stride=2, padding=0)
        self.derelu3 = nn.ReLU()
        self.decoder4 = nn.ConvTranspose2d(128, 32, kernel_size=2, stride=2, padding=0)
        self.derelu4 = nn.ReLU()
        self.decoder5 = nn.ConvTranspose2d(64, 3, kernel_size=2, stride=2, padding=0)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x1 = self.relu1(self.encoder1(x))
        # print(x1.shape)
        x2 = self.relu2(self.encoder2(x1))  
        # print(x2.shape)
        x3 = self.relu3(self.encoder3(x2))  
        # print(x3.shape)
        x4 = self.relu4(self.encoder4(x3))  
        # print(x4.shape)
        x5 = self.relu5(self.encoder5(x4))  
        # print(x5.shape)

        d1 = self.derelu1(self.decoder1(x5))
        # print(d1.shape)
        d2 = self.derelu2(self.decoder2(torch.concatenate([d1,self.drop1(x4)], dim=1)))
        # print(d2.shape)
        d3 = self.derelu3(self.decoder3(torch.concatenate([d2,self.drop2(x3)], dim=1)))
        # print(d3.shape)
        d4 = self.derelu4(self.decoder4(torch.concatenate([d3,self.drop3(x2)], dim=1)))
        # print(d4.shape)
        decoded = self.sigmoid(self.decoder5(torch.concatenate([d4 ,self.drop4(x1)], dim=1)))
        # print(decoded.shape)
        x = decoded 

        return x
