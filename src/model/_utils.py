import functools
from fastai.vision.all import *
import fastai

def rsetattr(obj, attr, val):
    pre, _, post = attr.rpartition('.')
    return setattr(rgetattr(obj, pre) if pre else obj, post, val)


def rgetattr(obj, attr, *args):
    def _getattr(obj, attr):
        return getattr(obj, attr, *args)

    return functools.reduce(_getattr, [obj] + attr.split('.'))


def get_layer_name(layer, idx):
    # TODO: minimal implementation based on class name + idx
    # type_str = str(type(layer))
    # type_str = type_str.split('.')[1][:-2]
    # return f"{type_str}_{idx}"

    if isinstance(layer, torch.nn.Conv2d):
        layer_name = 'Conv2d_{}_{}x{}'.format(
            idx, layer.in_channels, layer.out_channels
        )
    elif isinstance(layer, torch.nn.ConvTranspose2d):
        layer_name = 'ConvT2d_{}_{}x{}'.format(
            idx, layer.in_channels, layer.out_channels
        )
    elif isinstance(layer, torch.nn.BatchNorm2d):
        layer_name = 'BatchNorm2D_{}_{}'.format(
            idx, layer.num_features)
    elif isinstance(layer, torch.nn.Linear):
        layer_name = 'Linear_{}_{}x{}'.format(
            idx, layer.in_features, layer.out_features
        )
    elif isinstance(layer, fastai.layers.Identity):
        layer_name = 'Identity'
    else:
        layer_name = "Activation_{}".format(idx)
    # idx += 1
    # return layer_name, idx
    return '_'.join(layer_name.split('_')[:2]).lower()