#  #!/usr/bin/env python3
#  -*- coding: utf-8 -*-
#  Copyright (c) 2021.  Luca Clissa
#  #Licensed under the Apache License, Version 2.0 (the "License");
#  #you may not use this file except in compliance with the License.
#  #You may obtain a copy of the License at
#  #http://www.apache.org/licenses/LICENSE-2.0
#  #Unless required by applicable law or agreed to in writing, software
#  #distributed under the License is distributed on an "AS IS" BASIS,
#  #WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  #See the License for the specific language governing permissions and
#  #limitations under the License.
__all__ = ['_get_ltype', 'Add', 'Concatenate', 'ConvBlock', 'ResidualBlock', 'UpResidualBlock', 'Bottleneck',
           'Heatmap', 'Heatmap2d']

from fastai.vision.all import *
from ._utils import *

# Utils
def _get_ltype(layer):
    name = str(layer.__class__).split("'")[1]
    return name.split('.')[-1]

    # Blocks
class Add(nn.Module):
    def __init__(self):
        super(Add, self).__init__()
        self.add = torch.add

    def forward(self, x1, x2):
        return self.add(x1, x2)

class Concatenate(nn.Module):
    def __init__(self, dim):
        super(Concatenate, self).__init__()
        self.cat = partial(torch.cat, dim=dim)

    def forward(self, x):
        return self.cat(x)

# TODO: commented parts should allow blocks numbering. Find a way to do it automatically and propagate to different blocks
class ConvBlock(nn.Module):
    def __init__(self, n_in, n_out, kernel_size=3, stride=1, padding=1):  # , idb=1):
        super(ConvBlock, self).__init__()

        layers = [
            nn.BatchNorm2d(n_in, momentum=0.01, eps=0.001),
            nn.ELU(),
            nn.Conv2d(n_in, n_out, kernel_size, stride, padding),
            nn.BatchNorm2d(n_out, momentum=0.01, eps=0.001),
            nn.ELU(),
            nn.Conv2d(n_out, n_out, kernel_size, stride, padding),
        ]
        # self.idb = idb
        self._init_block(layers)

    def forward(self, x):
        for layer in self.conv_block.values():
            x = layer(x)
        return x

    def _init_block(self, layers):
        # self.add_module(f"conv_block{self.idb}", nn.ModuleDict())
        self.conv_block = nn.ModuleDict()
        for idx, layer in enumerate(layers):
            self.conv_block.add_module(get_layer_name(layer, idx), layer)
            # getattr(self, f"conv_block{self.idb}")[get_layer_name(layer, idx)] = layer

class IdentityPath(nn.Module):
    def __init__(self, n_in: int, n_out: int, is_conv: bool = True, upsample: bool = False):
        super(IdentityPath, self).__init__()

        self.is_conv = is_conv
        self.upsample = upsample

        # TODO:
        #  1) find elegant way to deal with is_conv=False, upsample=True; currently returns ConvT2d
        #  2) implement up_conv + concatenate directly here
        if upsample:
            layer = nn.ConvTranspose2d(n_in, n_out, kernel_size=2, stride=2, padding=0)
        elif is_conv:
            layer = nn.Conv2d(n_in, n_out, kernel_size=1, padding=0)
        else:
            layer = Identity()
        self.layer_name = get_layer_name(layer, 1)
        self.add_module(self.layer_name, layer)

    def forward(self, x):
        return getattr(self, self.layer_name)(x)

class ResidualBlock(nn.Module):
    def __init__(self, n_in, n_out, kernel_size=3, stride=1, padding=1, is_conv=True):
        super(ResidualBlock, self).__init__()

        self.is_conv = is_conv
        self.conv_path = ConvBlock(n_in, n_out, kernel_size=kernel_size, stride=stride, padding=padding)
        self.id_path = IdentityPath(n_in, n_out, is_conv=is_conv)
        self.add = Add()

    def forward(self, x):
        conv_path = self.conv_path(x)
        short_connect = self.id_path(x)
        return self.add(conv_path, short_connect)

class Bottleneck(nn.Module):
    def __init__(self, n_in, n_out, kernel_size=5, stride=1, padding=2):
        super(Bottleneck, self).__init__()

        self.residual_block1 = ResidualBlock(n_in, n_out, kernel_size=kernel_size, stride=stride, padding=padding,
                                             is_conv=True)
        self.residual_block2 = ResidualBlock(n_out, n_out, kernel_size=kernel_size, stride=stride, padding=padding,
                                             is_conv=False)

    def forward(self, x):
        return self.residual_block2(self.residual_block1(x))

class UpResidualBlock(nn.Module):
    def __init__(self, n_in, n_out, kernel_size=3, stride=1, padding=1, concat_dim=1):
        super(UpResidualBlock, self).__init__()
        self.id_path = nn.ModuleDict({
            "up_conv": nn.ConvTranspose2d(n_in, n_out, kernel_size=2, stride=2, padding=0),
            "concat": Concatenate(dim=concat_dim)
        })
        self.conv_path = ConvBlock(
            n_in, n_out, kernel_size=kernel_size, stride=stride, padding=padding)
        self.add = Add()

    def forward(self, x, long_connect):
        short_connect = self.id_path.up_conv(x)
        concat = self.id_path.concat([short_connect, long_connect])
        return self.add(self.conv_path(concat), short_connect)

class Heatmap(nn.Module):
    def __init__(self, n_in, n_out=1, kernel_size=1, stride=1, padding=0):
        super(Heatmap, self).__init__()
        self.conv2d = nn.Conv2d(n_in, n_out, kernel_size, stride, padding)
        self.act = nn.Sigmoid()

    def forward(self, x):
        return self.act(self.conv2d(x))

class Heatmap2d(nn.Module):
    def __init__(self, n_in, n_out=2, kernel_size=1, stride=1, padding=0, concat_dim=1):
        super(Heatmap2d, self).__init__()
        self.heatmap = Heatmap(n_in, n_out - 1, kernel_size, stride, padding)
        self.concat = Concatenate(dim=concat_dim)

    def forward(self, x):
        heatmap1 = self.heatmap(x)
        heatmap0 = torch.ones_like(heatmap1) - heatmap1
        return self.concat([heatmap0, heatmap1])