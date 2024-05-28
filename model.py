#import lightning as L
from torch import nn
import os
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
import numpy as np
import logging
from tensorflow.keras.losses import BinaryCrossentropy

class Models(nn.Module):
    def __init__(self, input_shape, output_shape):
        super().__init__()
        self.input_shape = input_shape
        self.output_shape = output_shape
        self.debug = True

class DownBlock(nn.Module):

    def __init__(self, in_channels: int, out_channels: int, stride: int):
        super().__init__()
        self.down = nn.Sequential(
            nn.Conv2d(in_channels, out_channels,
                      kernel_size=3, stride=stride, padding=1),
            nn.ReLU(),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(out_channels)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.down(x)
        return x

class UpBlock(nn.Module):

    def __init__(self, in_channels_up: int, in_channels_skip, out_channels: int):
        super().__init__()
        self.up_convtranspose = nn.Sequential(
            nn.ConvTranspose2d(in_channels_up, in_channels_up // 2,
                               kernel_size=3, stride=2, padding=0),
            nn.ReLU()
        )
        self.up_crb = nn.Sequential(
            nn.Conv2d(in_channels_up // 2 + in_channels_skip,
                      out_channels, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(out_channels)
        )

    def _crop(self, x, target_shape: torch.Tensor.shape) -> torch.Tensor:
        x = x[:, :, :target_shape[2], :target_shape[3], :target_shape[4]]

        return x

    def forward(self, x_skip, x_up):
        x_up = self.up_convtranspose(x_up)
        x_up = self._crop(x_up, x_skip.shape)
        x_up = torch.concat([x_skip, x_up], dim=1)
        x_up = self.up_crb(x_up)

        return x_up

class UNet2D(Models):
    
    def __init__(self, input_shape_with_channels, output_classes, depth, width):

        input_shape = input_shape_with_channels[1:] # x, y, z
        input_channels = input_shape_with_channels[0]

        # change to z, y, x for NP
        input_shape = input_shape[::-1]
        output_shape = [output_classes, input_shape[0], input_shape[1], input_shape[2]] 
                                                                                                                
        super().__init__(input_shape=input_shape, output_shape=output_shape)

        print('initializing model')
        self.input_channels = input_channels
        self.output_classes = output_classes
        layers = 6
        self.depth = layers
        self.width = width

        self.downblocks = nn.ModuleList()

        for i in range(self.depth):
            if i == 0:
                downblock_input_channels = input_channels
                downblock_output_channels = width * (2**(i + 2))
                stride = 1
            else:
                downblock_input_channels = width * (2**(i + 1))
                downblock_output_channels = width * (2**(i + 2))
                stride = 2
            if self.debug:
                print(
                    f"downblock_{i}, in: {downblock_input_channels}, out: {downblock_output_channels}, stride: {stride}")
            self.downblocks.append(DownBlock(
                in_channels=downblock_input_channels, out_channels=downblock_output_channels, stride=stride))

        self.upblocks = nn.ModuleList()
        for i in range(self.depth-1):
            in_channels_skip = width * (2**(i + 2))
            in_channels_up = width * (2**(i + 3))
            out_channels = width * (2**(i + 2))
            if self.debug:
                print(
                    f"upblock_{i}, in_channels_skip: {in_channels_skip}, in_channels_up: {in_channels_up}, out_channels: {out_channels}")
            self.upblocks.append(UpBlock(
                in_channels_skip=in_channels_skip,
                in_channels_up=in_channels_up,
                out_channels=out_channels
            ))

        self.outconv = nn.Conv3d(in_channels=(
            width * (4)), out_channels=output_classes, kernel_size=3, stride=1, padding=1)

    def forward(self, x):

        x_skip = []
        for i in range(self.depth):
            x = self.downblocks[i](x)
            x_skip.append(x)

        for i in range(self.depth-2, -1, -1):
            x = self.upblocks[i](x_skip=x_skip[i], x_up=x)

        x = self.outconv(x)

        return x
    

