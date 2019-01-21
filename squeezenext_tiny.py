# -*- coding: utf-8 -*-
import collections

import chainer
import chainer.functions as F
import chainer.links as L


class BasicBlock(chainer.Chain):

    def __init__(self, in_ch, out_ch, stride):
        super(BasicBlock, self).__init__()
        reduction = 0.5
        self.stride = stride
        self.in_ch  = in_ch
        self.out_ch = out_ch
        if stride == 2:
            reduction = 1
        elif in_ch > out_ch:
            reduction = 0.25

        with self.init_scope():
            self.conv1 = L.Convolution2D(in_ch, int(in_ch * reduction),
                                         ksize=1, stride=stride, nobias=False)
            self.bn1   = L.BatchNormalization(int(in_ch * reduction))
            self.conv2 = L.Convolution2D(int(in_ch * reduction), int(in_ch * reduction * 0.5),
                                         ksize=1, stride=1, nobias=False)
            self.bn2   = L.BatchNormalization(int(in_ch * reduction * 0.5))
            self.conv3 = L.Convolution2D(int(in_ch * reduction * 0.5), int(in_ch * reduction),
                                         ksize=(1, 3), stride=1, pad=(0, 1), nobias=False)
            self.bn3   = L.BatchNormalization(int(in_ch * reduction))
            self.conv4 = L.Convolution2D(int(in_ch * reduction), int(in_ch * reduction),
                                         ksize=(3, 1), stride=1, pad=(1, 0), nobias=False)
            self.bn4   = L.BatchNormalization(int(in_ch * reduction))
            self.conv5 = L.Convolution2D(int(in_ch * reduction), out_ch,
                                         ksize=1, stride=1, nobias=False)
            self.bn5   = L.BatchNormalization(out_ch)

            self.shortcut = chainer.Sequential(lambda x: x)
            if stride == 2 or in_ch != out_ch:
                self.shortcut.append(L.Convolution2D(in_ch, out_ch, ksize=1, stride=stride, nobias=False))
                self.shortcut.append(L.BatchNormalization(out_ch))

    def __call__(self, x):
        h = F.relu(self.bn1(self.conv1(x)))
        h = F.relu(self.bn2(self.conv2(h)))
        h = F.relu(self.bn3(self.conv3(h)))
        h = F.relu(self.bn4(self.conv4(h)))
        h = F.relu(self.bn5(self.conv5(h)))
        h += F.relu(self.shortcut(x))
        return F.relu(h)


class SqueezeNext(chainer.Chain):

    #insize = 32  # for Cifar
    insize = 64  # for Tiny Imagenet

    def __init__(self, width_x, blocks, num_classes, pretrained_model=None):
        super(SqueezeNext, self).__init__()
        self.in_ch = 64
        with self.init_scope():
            #self.conv1 = L.Convolution2D(3, int(width_x * self.in_ch), ksize=3, stride=1, pad=1)  # for Cifar
            self.conv1  = L.Convolution2D(3, int(width_x * self.in_ch), ksize=3, stride=2, pad=1)  # for Tiny Imagenet
            self.bn1    = L.BatchNormalization(int(width_x * self.in_ch))
            self.stage1 = self._make_layers(blocks[0], width_x, 32, 1)
            self.stage2 = self._make_layers(blocks[1], width_x, 64, 2)
            self.stage3 = self._make_layers(blocks[2], width_x, 128, 2)
            self.stage4 = self._make_layers(blocks[3], width_x, 256, 2)
            self.conv2  = L.Convolution2D(int(width_x * self.in_ch), int(width_x * 128),
                                          ksize=1, stride=1, nobias=False)
            self.bn2    = L.BatchNormalization(int(width_x * 128))
            self.fc     = L.Linear(int(width_x * 128), num_classes)

        if pretrained_model is not None:
            chainer.serializers.load_npz(pretrained_model, self)

    def _make_layers(self, num_block, width_x, out_ch, stride):
        strides = [stride] + [1] * (num_block -1)
        layers  = chainer.Sequential()
        for _stride in strides:
            layers.append(BasicBlock(int(width_x * self.in_ch), int(width_x * out_ch), _stride))
            self.in_ch = out_ch
        return layers

    @property
    def functions(self):
        return collections.OrderedDict([
            ('conv1',  [self.conv1, self.bn1, F.relu]),
            ('stage1', [self.stage1]),
            ('stage2', [self.stage2]),
            ('stage3', [self.stage3]),
            ('stage4', [self.stage4]),
            ('conv2',  [self.conv2, self.bn2, F.relu]),
            ('pool',   [_global_average_pooling_2d]),
            #('pool',   [lambda x: F.average_pooling_2d(x, 4)]),  # for Cifar and Tiny ImageNet img size
            ('prob',     [self.fc])
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

    def __call__(self, x):
        return self.forward(x)


def _global_average_pooling_2d(x):
    n, channel, rows, cols = x.shape
    h = F.average_pooling_2d(x, (rows, cols), stride=1)
    h = F.reshape(h, (n, channel))
    return h


class SqNxt_23_1x(SqueezeNext):

    def __init__(self, num_classes=1000, pretrained_model=None):
        width_x = 1.0
        blocks  = [6, 6, 8, 1]
        super(SqNxt_23_1x, self).__init__(width_x, blocks, num_classes, pretrained_model)


class SqNxt_23_1x_v5(SqueezeNext):

    def __init__(self, num_classes=1000, pretrained_model=None):
        width_x = 1.0
        blocks  = [2, 4, 14, 1]
        super(SqNxt_23_1x_v5, self).__init__(width_x, blocks, num_classes, pretrained_model)


class SqNxt_23_2x(SqueezeNext):

    def __init__(self, num_classes=1000, pretrained_model=None):
        width_x = 2.0
        blocks  = [6, 6, 8, 1]
        super(SqNxt_23_2x, self).__init__(width_x, blocks, num_classes, pretrained_model)


class SqNxt_23_2x_v5(SqueezeNext):

    def __init__(self, num_classes=1000, pretrained_model=None):
        width_x = 2.0
        blocks  = [2, 4, 14, 1]
        super(SqNxt_23_2x_v5, self).__init__(width_x, blocks, num_classes, pretrained_model)
