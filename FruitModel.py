import torch
import torch.nn as nn
import math


class FruitModel(nn.Module):

    def __init__(self, image_size=100, in_channels=3):

        super(FruitModel, self).__init__()

        self.conv1 = self._create_conv_layer_pool(
            in_channels=in_channels, out_channels=16)
        self.conv2 = self._create_conv_layer_pool(
            in_channels=16, out_channels=32)
        self.conv3 = self._create_conv_layer_pool(
            in_channels=32, out_channels=64)
        self.conv4 = self._create_conv_layer_pool(
            in_channels=64, out_channels=64)
        self.conv5 = self._create_conv_layer_pool(
            in_channels=64, out_channels=64)

        in_features = 3*3*64

        self.linear1 = self._create_linear_layer(in_features, 256)
        self.linear2 = self._create_linear_layer(256, 131)

    def forward(self, x):

        batch_size, channels, width, height = x.shape
        x = x.view(-1, channels, width, height)

        # CNN Layers
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)

        x = x.view(batch_size, -1)

        # Linear Layers
        x = self.linear1(x)
        x = self.linear2(x)

        return x

    def _create_linear_layer(self, in_features, out_features, p=0.6):
        return nn.Sequential(
            nn.Linear(in_features, out_features),
            nn.Dropout(p=p)
        )

    def _create_conv_layer(self, in_channels, out_channels, kernel_size=(3, 3), padding=(1, 1)):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels,
                      kernel_size, padding=padding),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
        )

    def _create_conv_layer_pool(self, in_channels, out_channels, kernel_size=(3, 3), padding=(1, 1), pool=(2, 2)):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels,
                      kernel_size, padding=padding),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.MaxPool2d(pool)
        )
