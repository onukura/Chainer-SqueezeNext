# -*- coding: utf-8 -*-

import sys
import os
import collections

import chainer
import chainer.functions as F
import chainer.links as L
from chainer.dataset import download
from chainer.serializers import npz


class BasicBlock(chainer.Chain):

    _con3_4_types = {
        'a': {'conv3_ksize': (3, 1), 'conv3_pad': (1, 0),
              'conv4_ksize': (1, 3), 'conv4_pad': (0, 1)},
        'b': {'conv3_ksize': (1, 3), 'conv3_pad': (0, 1),
              'conv4_ksize': (3, 1), 'conv4_pad': (1, 0)},
    }
    
    def __init__(self, in_ch, out_ch, stride, con3_4_type):
        super(BasicBlock, self).__init__()
        reduction = 0.5
        self.stride = stride
        self.in_ch  = in_ch
        self.out_ch = out_ch
        if stride == 2:
            reduction = 1
            self.resize_identity = True
        elif in_ch > out_ch:
            reduction = 0.25
            self.resize_identity = True
        else:
            reduction = 0.5
            self.resize_identity = False

        with self.init_scope():
            self.conv1 = L.Convolution2D(in_ch, int(in_ch * reduction),
                                         ksize=1, stride=stride, nobias=True)
            self.bn1   = L.BatchNormalization(int(in_ch * reduction))
            self.conv2 = L.Convolution2D(int(in_ch * reduction), int(in_ch * reduction * 0.5),
                                         ksize=1, stride=1, nobias=True)
            self.bn2   = L.BatchNormalization(int(in_ch * reduction * 0.5))
            self.conv3 = L.Convolution2D(int(in_ch * reduction * 0.5), int(in_ch * reduction),
                                         ksize=self._con3_4_types[con3_4_type]['conv3_ksize'], stride=1,
                                         pad=self._con3_4_types[con3_4_type]['conv3_pad'], nobias=True)
            self.bn3   = L.BatchNormalization(int(in_ch * reduction))
            self.conv4 = L.Convolution2D(int(in_ch * reduction), int(in_ch * reduction),
                                         ksize=self._con3_4_types[con3_4_type]['conv4_ksize'], stride=1,
                                         pad=self._con3_4_types[con3_4_type]['conv4_pad'], nobias=True)
            self.bn4   = L.BatchNormalization(int(in_ch * reduction))
            self.conv5 = L.Convolution2D(int(in_ch * reduction), out_ch,
                                         ksize=1, stride=1, nobias=True)
            self.bn5   = L.BatchNormalization(out_ch)

            self.shortcut = chainer.Sequential(lambda x: x)
            if self.resize_identity:
                self.shortcut.append(L.Convolution2D(in_ch, out_ch,
                                                     ksize=1, stride=stride, nobias=True))
                self.shortcut.append(L.BatchNormalization(out_ch))

    def __call__(self, x):
        h = F.relu(self.bn1(self.conv1(x)))
        h = F.relu(self.bn2(self.conv2(h)))
        h = F.relu(self.bn3(self.conv3(h)))
        h = F.relu(self.bn4(self.conv4(h)))
        h = F.relu(self.bn5(self.conv5(h)))
        h += F.relu(self.shortcut(x))
        return F.relu(h)


class InitialBlock(chainer.Chain):

    def __init__(self, in_ch, out_ch):
        super(InitialBlock, self).__init__()
        with self.init_scope():
            self.conv = L.Convolution2D(in_ch, out_ch, ksize=7, stride=2, pad=1, nobias=True)

    def __call__(self, x):
        h = self.conv(x)
        return F.max_pooling_2d(h, ksize=3, stride=2)


class SqueezeNext(chainer.Chain):    
    
    insize = (224, 224)
    
    def __init__(self, channels, init_block_ch, final_block_ch, in_ch=3, 
                 num_classes=1000, pretrained_model=None):
        self.num_classes = num_classes
        super(SqueezeNext, self).__init__()
        with self.init_scope():
            self.init_block = InitialBlock(in_ch=in_ch, out_ch=init_block_ch)
            in_ch = init_block_ch
            con3_4_type = 'a'
            for i, ch_per_stage in enumerate(channels):
                stage = chainer.Sequential()
                for j, out_ch in enumerate(ch_per_stage):
                    stride = 2 if (j == 0) and (i != 0) else 1
                    stage.append(BasicBlock(in_ch=in_ch, out_ch=out_ch, stride=stride, con3_4_type=con3_4_type))
                    in_ch = out_ch
                    if con3_4_type == 'a':
                        con3_4_type = 'b'
                    elif con3_4_type == 'b':
                        con3_4_type = 'a'
                setattr(self, 'stage{}'.format(i + 1), stage)
            self.final_block = L.Convolution2D(in_ch, final_block_ch,
                                               ksize=1, stride=1, nobias=True)
            in_ch = final_block_ch
            self.fc = L.Linear(in_ch, num_classes)
            
        if pretrained_model is not None:
            chainer.serializers.load_npz(pretrained_model, self)

    @property
    def functions(self):        
        return collections.OrderedDict([
            ('init_block',  [self.init_block]),
            ('stage1',      [self.stage1]),
            ('stage2',      [self.stage2]),
            ('stage3',      [self.stage3]),
            ('stage4',      [self.stage4]),
            ('final_block', [self.final_block]),
            ('final_pool',  [lambda x: _global_average_pooling_2d(x)]),
            ('prob',        [self.fc])
        ])

    @property
    def available_layers(self):
        return list(self.functions.keys())

    def forward(self, x, layers=None):
        """forward(self, x, layers=['prob'])
        Computes all the feature maps specified by ``layers``.
        Args:
            x (~chainer.Variable): Input variable. It should be prepared by
                ``prepare`` function.
            layers (list of str): The list of layer names you want to extract.
                If ``None``, 'prob' will be used as layers.
        Returns:
            Dictionary of ~chainer.Variable: A dictionary in which
            the key contains the layer and the value contains the
            corresponding feature map variable.
        """

        if layers is None:
            layers = ['prob']

        h = x
        activations = {}
        target_layers = set(layers)
        for key, funcs in self.functions.items():
            if len(target_layers) == 0:
                break
            for func in funcs:
                h = func(h)
            if key in target_layers:
                activations[key] = h
                target_layers.remove(key)
        return activations

    def __call__(self, x, layers=None):
        return self.forward(x, layers)


class SqueezeNext23_W1(SqueezeNext):
    """
    1.0-SqNxt-23 model from 'SqueezeNext: Hardware-Aware Neural Network Design,' https://arxiv.org/abs/1803.10615.
    """
    
    def __init__(self):
        channels, init_block_ch, final_block_ch = _get_parameter(ver='23_w1')
        super(SqueezeNext23_W1, self).__init__(
                channels, init_block_ch, final_block_ch, in_ch=3,
                num_classes=1000, pretrained_model=None)
    

class SqueezeNext23_W3D2(SqueezeNext):
    """
    0.75-SqNxt-23 model from 'SqueezeNext: Hardware-Aware Neural Network Design,' https://arxiv.org/abs/1803.10615.
    """    
    
    def __init__(self):
        channels, init_block_ch, final_block_ch = _get_parameter(ver='23_w3d2')
        super(SqueezeNext23_W3D2, self).__init__(
                channels, init_block_ch, final_block_ch, in_ch=3,
                num_classes=1000, pretrained_model=None)
    
    
class SqueezeNext23_W2(SqueezeNext):
    """
    0.5-SqNxt-23 model from 'SqueezeNext: Hardware-Aware Neural Network Design,' https://arxiv.org/abs/1803.10615.
    """
        
    def __init__(self):
        channels, init_block_ch, final_block_ch = _get_parameter(ver='23_w2')
        super(SqueezeNext23_W2, self).__init__(
                channels, init_block_ch, final_block_ch, in_ch=3,
                num_classes=1000, pretrained_model=None)


class SqueezeNext23v5_W1(SqueezeNext):
    """
    1.0-SqNxt-23v5 model from 'SqueezeNext: Hardware-Aware Neural Network Design,' https://arxiv.org/abs/1803.10615.
    """
        
    def __init__(self):
        channels, init_block_ch, final_block_ch = _get_parameter(ver='23v5_w1')
        super(SqueezeNext23v5_W1, self).__init__(
                channels, init_block_ch, final_block_ch, in_ch=3,
                num_classes=1000, pretrained_model=None)


class SqueezeNext23v5_W3D2(SqueezeNext):
    """
    0.75-SqNxt-23v5 model from 'SqueezeNext: Hardware-Aware Neural Network Design,' https://arxiv.org/abs/1803.10615.
    """
        
    def __init__(self):
        channels, init_block_ch, final_block_ch = _get_parameter(ver='23v5_w3d2')
        super(SqueezeNext23v5_W3D2, self).__init__(
                channels, init_block_ch, final_block_ch, in_ch=3,
                num_classes=1000, pretrained_model=None)
        

class SqueezeNext23v5_W2(SqueezeNext):
    """
    0.5-SqNxt-23v5 model from 'SqueezeNext: Hardware-Aware Neural Network Design,' https://arxiv.org/abs/1803.10615.
    """
        
    def __init__(self):
        channels, init_block_ch, final_block_ch = _get_parameter(ver='23v5_w2')
        super(SqueezeNext23v5_W2, self).__init__(
                channels, init_block_ch, final_block_ch, in_ch=3,
                num_classes=1000, pretrained_model=None)


def _update_block_ch(channels, init_block_ch, final_block_ch, width_scale):
    if width_scale != 1:
        channels = [[int(cij * width_scale) for cij in ci] for ci in channels]
        init_block_ch = int(init_block_ch * width_scale)
        final_block_ch = int(final_block_ch * width_scale)
    return channels, init_block_ch, final_block_ch


def _get_parameter(ver):
    init_block_ch = 64
    final_block_ch = 128
    channels_per_layers = [32, 64, 128, 256]
    if ver == '23_w1':
        layers = [6, 6, 8, 1]
        width_scale = 1.0
    elif ver == '23_w3d2':
        layers = [6, 6, 8, 1]
        width_scale = 1.5        
    elif ver == '23_w2':
        layers = [6, 6, 8, 1]
        width_scale = 2.0  
    elif ver == '23v5_w1':
        layers = [2, 4, 14, 1]
        width_scale = 1.0
    elif ver == '23v5_w3d2':
        layers = [2, 4, 14, 1]
        width_scale = 1.5        
    elif ver == '23v5_w2':
        layers = [2, 4, 14, 1]
        width_scale = 2.0  
    channels = [[ci] * li for (ci, li) in zip(channels_per_layers, layers)]
    if width_scale != 1:
        channels = [[int(cij * width_scale) for cij in ci] for ci in channels]
        init_block_ch = int(init_block_ch * width_scale)
        final_block_ch = int(final_block_ch * width_scale)
    return channels, init_block_ch, final_block_ch

     
def _global_average_pooling_2d(x):
    n, channel, rows, cols = x.shape
    h = F.average_pooling_2d(x, (rows, cols), stride=1)
    h = F.reshape(h, (n, channel))
    return h
