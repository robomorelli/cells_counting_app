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
__all__ = ['ResUnet', 'c_resunet', 'load_model']

from fastai.vision.all import *
from ._blocks import *
from ._utils import *
import torch.nn as nn

class ResUnet(nn.Module):
    def __init__(self, n_features_start=16, n_out=1):
        super(ResUnet, self).__init__()
        pool_ks, pool_stride, pool_pad = 2, 2, 0

        self.encoder = nn.ModuleDict({
            'colorspace': nn.Conv2d(3, 1, kernel_size=1, padding=0),

            # block 1
            'conv_block': ConvBlock(1, n_features_start),
            'pool1': nn.MaxPool2d(pool_ks, pool_stride, pool_pad),

            # block 2
            'residual_block1': ResidualBlock(n_features_start, 2 * n_features_start, is_conv=True),
            'pool2': nn.MaxPool2d(pool_ks, pool_stride, pool_pad),

            # block 3
            'residual_block2': ResidualBlock(2 * n_features_start, 4 * n_features_start, is_conv=True),
            'pool3': nn.MaxPool2d(pool_ks, pool_stride, pool_pad),

            # bottleneck
            'bottleneck': Bottleneck(4 * n_features_start, 8 * n_features_start, kernel_size=5, padding=2),
        })

        self.decoder = nn.ModuleDict({
            # block 6
            'upconv_block1': UpResidualBlock(n_in=8 * n_features_start, n_out=4 * n_features_start),

            # block 7
            'upconv_block2': UpResidualBlock(4 * n_features_start, 2 * n_features_start),

            # block 8
            'upconv_block3': UpResidualBlock(2 * n_features_start, n_features_start),
        })

        # output
        #self.head = Heatmap2d(
        #    n_features_start, n_out, kernel_size=1, stride=1, padding=0)
        self.head = Heatmap(
            n_features_start, n_out, kernel_size=1, stride=1, padding=0)

    def _forward_impl(self, x: Tensor) -> Tensor:
        downblocks = []
        for lbl, layer in self.encoder.items():
            x = layer(x)
            if 'block' in lbl: downblocks.append(x)
            # NEXT loop is hon the values and so we don't hane the name as in the items of the previous loop
        for layer, long_connect in zip(self.decoder.values(), reversed(downblocks)):
            x = layer(x, long_connect)
        return self.head(x)

    def init_kaiming_normal(self, mode='fan_in'):
        print('Initializing conv2d weights with Kaiming He normal')
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                m.weight = nn.init.kaiming_normal_(m.weight, mode=mode)
            elif isinstance(m, nn.BatchNorm3d) or isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def forward(self, x: Tensor) -> Tensor:
        return self._forward_impl(x)


def _resunet(
        arch: str,
        n_features_start: int,
        n_out: int,
        #     block: Type[Union[BasicBlock, Bottleneck]],
        #     layers: List[int],
        pretrained: bool,
        progress: bool,
        **kwargs,
) -> ResUnet:
    model = ResUnet(n_features_start, n_out)  # , **kwargs)
    model.__name__ = arch
    # TODO: implement weights fetching if not present
    if pretrained:
        print('no pretraine, stra from zero')
        model.init_kaiming_normal()

    else:
        model.init_kaiming_normal()
    return model

def c_resunet(arch='c-ResUnet', n_features_start: int = 16, n_out: int = 1, pretrained: bool = False,
              progress: bool = True,
              **kwargs) -> ResUnet:
    r"""cResUnet model from `"Automating Cell Counting in Fluorescent Microscopy through Deep Learning with c-ResUnet"
    <https://www.nature.com/articles/s41598-021-01929-5>`_.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _resunet(arch=arch, n_features_start=n_features_start, n_out=n_out, pretrained=pretrained,
                    progress=progress, **kwargs)


def load_model(resume_path, device, n_features_start=16, n_out=1, fine_tuning=False
               ,unfreezed_layers=1):


    model = nn.DataParallel(c_resunet(arch='c-ResUnet', n_features_start=n_features_start, n_out=n_out,
                                      device=device).to(device))

    checkpoint_file = torch.load(resume_path)
    model.load_state_dict(checkpoint_file, strict=False)
    if fine_tuning:
        print('fine_tuning')
        #if unfreezed_layers.isdecimal():
        if len(unfreezed_layers) == 1:
            unfreezed_layers = int(unfreezed_layers[0])
            for block in list(list(model.children())[0].named_children())[::-1]:  # encoder, decoder, head
                # print('unfreezing {} of {}'.format(unfreezed_layers, block))
                if block[0] == 'head':
                    for nc, cc in list(block[1].named_children())[::-1]:  # [1] because 0 is the name
                        if unfreezed_layers > 0:  # and isinstance(c, nn.Conv2d):
                            for n, p in cc.named_parameters():
                                p.requires_grad_(True)
                                # print(block, n, p.requires_grad)
                            print('unfreezed {}'.format(nc))
                        else:
                            for n, p in cc.named_parameters():
                                p.requires_grad_(False)
                                # print(block, n, p.requires_grad)
                    unfreezed_layers = int(unfreezed_layers) - 1

                else:
                    for nc, cc in list(block[1].named_children())[::-1]:
                        if unfreezed_layers > 0:
                            for n, p in cc.named_parameters():
                                p.requires_grad_(True)
                            print('unfreezed {}'.format(nc))
                        else:
                            for n, p in cc.named_parameters():
                                p.requires_grad_(False)
                            print('keep freezed {}'.format(nc))
                        unfreezed_layers = int(unfreezed_layers) - 1

            print('requires grad for each layer:')
            for block in list(list(model.children())[0].named_children())[::-1]:
                for n, p in list(block[1].named_parameters())[::-1]:
                    print(n, p.requires_grad)

        elif len(unfreezed_layers) == 2 and int(unfreezed_layers[0]) == 0:
            unfreezed_layers_start = int(unfreezed_layers[0])
            unfreezed_layers_end = int(unfreezed_layers[1])

            unfreezed_layers = unfreezed_layers_end - unfreezed_layers_start

            for block in list(list(model.children())[0].named_children()):  # encoder, decoder, head
                #print('unfreezing {} of {}'.format(unfreezed_layers, block))
                if block[0] == 'head':
                    for nc, cc in list(block[1].named_children()):  # [1] because 0 is the name
                        if unfreezed_layers > 0: #and isinstance(c, nn.Conv2d):
                            for n, p in cc.named_parameters():
                                p.requires_grad_(True)
                                #print(block, n, p.requires_grad)
                            print('unfreezed {}'.format(nc))
                            unfreezed_layers = int(unfreezed_layers) - 1
                        else:
                            for n, p in cc.named_parameters():
                                p.requires_grad_(False)
                                #print(block, n, p.requires_grad)


                else:
                    for nc, cc in list(block[1].named_children()):
                        if unfreezed_layers > 0:
                            for n, p in cc.named_parameters():
                                p.requires_grad_(True)
                            print('unfreezed {}'.format(nc))
                        else:
                            for n, p in cc.named_parameters():
                                p.requires_grad_(False)
                            print('keep freezed {}'.format(nc))
                        unfreezed_layers = int(unfreezed_layers) - 1

            print('requires grad for each layer:')
            for block in list(list(model.children())[0].named_children()):
                for n, p in list(block[1].named_parameters()):
                    print(n, p.requires_grad)

        elif len(unfreezed_layers) == 2 and int(unfreezed_layers[0]) != 0:
            unfreezed_layers_start = int(unfreezed_layers[0])
            unfreezed_layers_end = int(unfreezed_layers[1])
            ixs_list = np.arange(unfreezed_layers_start, unfreezed_layers_end, 1)
            unfreezed_layers = unfreezed_layers_end - unfreezed_layers_start
            ix = 0
            for block in list(list(model.children())[0].named_children()):  # encoder, decoder, head
                #print('unfreezing {} of {}'.format(unfreezed_layers, block))
                    if block[0] == 'head':
                        for nc, cc in list(block[1].named_children()):
                            if ix in ixs_list:# [1] because 0 is the name
                                if unfreezed_layers > 0: #and isinstance(c, nn.Conv2d):
                                    for n, p in cc.named_parameters():
                                        p.requires_grad_(True)
                                        #print(block, n, p.requires_grad)
                                    print('unfreezed {}'.format(nc))
                                else:
                                    for n, p in cc.named_parameters():
                                        p.requires_grad_(False)
                                        #print(block, n, p.requires_grad)
                            else:
                                for n, p in cc.named_parameters():
                                    p.requires_grad_(False)
                                    # print(block, n, p.requires_grad)
                            ix += 1


                    else:
                        for nc, cc in list(block[1].named_children()):
                            if 'pool' not in nc:
                                if ix in ixs_list:
                                    if unfreezed_layers > 0:
                                        for n, p in cc.named_parameters():
                                            p.requires_grad_(True)
                                        print('unfreezed {}'.format(nc))
                                    else:
                                        for n, p in cc.named_parameters():
                                            p.requires_grad_(False)
                                        print('keep freezed {}'.format(nc))
                                    #unfreezed_layers = int(unfreezed_layers) - 1

                                else:
                                    for n, p in cc.named_parameters():
                                        p.requires_grad_(False)
                                    print('keep freezed {}'.format(nc))

                                ix += 1

            print('requires grad for each layer:')
            for block in list(list(model.children())[0].named_children()):
                for n, p in list(block[1].named_parameters()):
                    print(n, p.requires_grad)


        elif len(unfreezed_layers) == 2 and int(unfreezed_layers[0]) == int(unfreezed_layers[0]):
            unfreezed_layers_start = int(unfreezed_layers[0])
            unfreezed_layers_end = int(unfreezed_layers[1])

            unfreezed_layers = unfreezed_layers_end - unfreezed_layers_start

            for block in list(list(model.children())[0].named_children()):  # encoder, decoder, head
                #print('unfreezing {} of {}'.format(unfreezed_layers, block))
                if block[0] == 'head':
                    for nc, cc in list(block[1].named_children()):  # [1] because 0 is the name
                        if unfreezed_layers > 0: #and isinstance(c, nn.Conv2d):
                            for n, p in cc.named_parameters():
                                p.requires_grad_(True)
                                #print(block, n, p.requires_grad)
                            print('unfreezed {}'.format(nc))
                            unfreezed_layers = int(unfreezed_layers) - 1
                        else:
                            for n, p in cc.named_parameters():
                                p.requires_grad_(False)
                                #print(block, n, p.requires_grad)


                else:
                    for nc, cc in list(block[1].named_children()):
                        if unfreezed_layers > 0:
                            for n, p in cc.named_parameters():
                                p.requires_grad_(True)
                            print('unfreezed {}'.format(nc))
                        else:
                            for n, p in cc.named_parameters():
                                p.requires_grad_(False)
                            print('keep freezed {}'.format(nc))
                        unfreezed_layers = int(unfreezed_layers) - 1

            print('requires grad for each layer:')
            for block in list(list(model.children())[0].named_children()):
                for n, p in list(block[1].named_parameters()):
                    print(n, p.requires_grad)


        elif int(unfreezed_layers[0]) == int(unfreezed_layers[1]) and len(unfreezed_layers) ==3:


            unfreezed_layers_num = int(unfreezed_layers[0])
            last_layer = int(unfreezed_layers[-1])

            if last_layer not in [0,1]:
                raise Exception("last_layer is not 0 or 1")

            for block in list(list(model.children())[0].named_children()):  # encoder, decoder, head
                #print('unfreezing {} of {}'.format(unfreezed_layers, block))
                if block[0] == 'head':
                    for nc, cc in list(block[1].named_children()):  # [1] because 0 is the name
                        if last_layer: #and isinstance(c, nn.Conv2d):
                            for n, p in cc.named_parameters():
                                p.requires_grad_(True)
                                #print(block, n, p.requires_grad)
                            print('unfreezed {}'.format(nc))
                        else:
                            for n, p in cc.named_parameters():
                                p.requires_grad_(False)
                                #print(block, n, p.requires_grad)

                else:
                    for nc, cc in list(block[1].named_children()):
                        if str(unfreezed_layers_num) in nc:
                            for n, p in cc.named_parameters():
                                p.requires_grad_(True)
                            print('unfreezed {}'.format(nc))
                        else:
                            for n, p in cc.named_parameters():
                                p.requires_grad_(False)
                            print('keep freezed {}'.format(nc))

            print('requires grad for each layer:')
            for block in list(list(model.children())[0].named_children()):
                for n, p in list(block[1].named_parameters()):
                    print(n, p.requires_grad)


        elif int(unfreezed_layers[0]) == int(unfreezed_layers[1]) and len(unfreezed_layers) > 2:

            unfreezed_layers_names = []
            last_layer = int(unfreezed_layers.pop())
            ixs = []
            for ix, x in enumerate(unfreezed_layers):
                if not x.isdecimal():
                    unfreezed_layers_names.append(str(x))
                else:
                    ixs.append(ix)

            unfreezed_layers = [unfreezed_layers[ix] for ix in ixs]
            unfreezed_layers = np.unique(unfreezed_layers )

            dict_name = {'1': ['conv_block', 'upconv_block1', 'pool1'],
                         '2': ['residual_block1', 'upconv_block2', 'pool2'],
                         '3': ['residual_block2', 'upconv_block3', 'pool3']}
            try:
                unfreezed_layers_names_temp = [dict_name[x] for x in unfreezed_layers]
                for li in unfreezed_layers_names_temp:
                    unfreezed_layers_names.extend(li)
            except:
                print(' no numeric value')

            if last_layer not in [0, 1]:
                raise Exception("last_layer is not 0 or 1")

            for block in list(list(model.children())[0].named_children()):  # encoder, decoder, head
                # print('unfreezing {} of {}'.format(unfreezed_layers, block))
                if block[0] == 'head':
                    for nc, cc in list(block[1].named_children()):  # [1] because 0 is the name
                        if last_layer:  # and isinstance(c, nn.Conv2d):
                            for n, p in cc.named_parameters():
                                p.requires_grad_(True)
                                # print(block, n, p.requires_grad)
                            print('unfreezed {}'.format(nc))
                        else:
                            for n, p in cc.named_parameters():
                                p.requires_grad_(False)
                                # print(block, n, p.requires_grad)

                else:
                    for nc, cc in list(block[1].named_children()):
                        for x in unfreezed_layers_names:
                            if x.isdecimal():
                                if x in nc:
                                    for n, p in cc.named_parameters():
                                        p.requires_grad_(True)
                                    print('unfreezed {}'.format(nc))
                                    break
                                else:
                                    for n, p in cc.named_parameters():
                                        p.requires_grad_(False)
                                    print('keep freezed {}'.format(nc))
                            else:
                                if x == nc:
                                    for n, p in cc.named_parameters():
                                        p.requires_grad_(True)
                                    print('unfreezed {}'.format(nc))
                                    break
                                else:
                                    for n, p in cc.named_parameters():
                                        p.requires_grad_(False)
                                    print('keep freezed {}'.format(nc))

            print('requires grad for each layer:')
            for block in list(list(model.children())[0].named_children()):
                for n, p in list(block[1].named_parameters()):
                    print(n, p.requires_grad)

    return