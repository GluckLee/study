import time
import paddle.nn as nn
import math
from prepro_reader import *

class ConvBNLayer(paddle.nn.Layer):
    def __init__(self,
                 num_channels,
                 num_filters,
                 filter_size,
                 stride=1,
                 groups=1,
                 act=None):
        super(ConvBNLayer, self).__init__()
        self._conv = nn.Conv2D(
            in_channels=num_channels,
            out_channels=num_filters,
            kernel_size=filter_size,
            stride=stride,
            padding=(filter_size-1)//2,
            groups=groups,
            bias_attr=False)
        self._batch_norm = paddle.nn.BatchNorm2D(num_filters)
        self.act = act
    def forward(self, inputs):
        print('ConvBNLayer***********************************************:')
        y = self._conv(inputs)
        print(y.shape)
        y = self._batch_norm(y)
        if self.act == 'leaky':
            y = F.leaky_relu(x=y, negative_slope=0.1)
        elif self.act == 'relu':
            y = F.relu(x=y)
        print(y.shape)
        print('ConvBNLayer***********************************************:')
        return y

class BottleneckBlock(paddle.nn.Layer):
    def __init__(self,
                 num_channels,
                 num_filters,
                 stride,
                 shortcut=True ):
        super(BottleneckBlock, self).__init__()
        self.conv0 = ConvBNLayer(
            num_channels=num_channels,
            num_filters=num_filters,
            filter_size=1,
            act='relu')
        self.conv1 = ConvBNLayer(
            num_channels=num_filters,
            num_filters=num_filters,
            filter_size=3,
            stride=stride,
            act='relu')
        self.conv2 = ConvBNLayer(
            num_channels=num_filters,
            num_filters=num_filters*4,
            filter_size=1,
            act=None)
        if not shortcut:
            self.short = ConvBNLayer(
                num_channels=num_channels,
                num_filters=num_filters*4,
                filter_size=1,
                stride=stride)
        self.shortcut = shortcut
        self._num_channels_out = num_filters*4

    def forward(self, inputs):
        print('BottleneckBlock###################################################:')
        y = self.conv0(inputs)
        print(y.shape)
        conv1 = self.conv1(y)
        print(conv1.shape)
        conv2 = self.conv2(conv1)
        print(conv2.shape)
        if self.shortcut:
            short = inputs
        else:
            short = self.short(inputs)
        print('--------->', short.shape, conv2.shape)
        y = paddle.add(x=short, y=conv2)
        print('--------->', y.shape)
        y = F.relu(y)
        print(y.shape)
        print('BottleneckBlock###################################################:')
        return y

class ResNet(paddle.nn.Layer):
    def __init__(self, layers=50, class_dim=1):
        super(ResNet, self).__init__()
        self.layers = layers
        supported_layers = [50, 101, 152]
        assert layers in supported_layers,\
            "supported layers are {} but input layer is {}".format(supported_layers, layers)
        if layers == 50:
            depth = [3, 4, 6, 3]
        elif layers == 101:
            depth = [3, 4, 23, 3]
        elif layers == 152:
            depth = [3, 8, 36, 3]

        num_filters = [64, 128, 256, 512]
        self.conv = ConvBNLayer(
            num_channels=3,
            num_filters=64,
            filter_size=7,
            stride=2,
            act='relu')
        self.pool2d_max = nn.MaxPool2D(
            kernel_size=3, stride=2, padding=1)

        self.bottleneck_block_list = []
        num_channels = 64
        # 遍历c2, c3, c4, c5
        for block in range(len(depth)):     # block:0，1，2，3
            shortcut = False
            # depth:[3, 4, 6, 3]
            for i in range(depth[block]):  # i 是【0,1,2】【0,1,2,3】【0,1,2,3,4,5】【0,1,2】
                bottleneck_block = self.add_sublayer(
                    'bb_%d_%d' % (block, i),
                    BottleneckBlock(
                        num_channels=num_channels,      # 64
                        num_filters=num_filters[block],     # key:block(0,1,2,3) --> [64, 128, 256, 512]
                        stride=2 if i == 0 and block != 0 else 1,   # c3, c4, c5 残差块 stride = 2, c2 用0
                        shortcut=shortcut))
                num_channels = bottleneck_block._num_channels_out
                self.bottleneck_block_list.append(bottleneck_block)
                shortcut = True
        self.pool2d_avg = paddle.nn.AdaptiveAvgPool2D(output_size=1)

        stdv = 1.0/math.sqrt(2048*1.0)
        self.out = nn.Linear(in_features=2048, out_features=class_dim,
                             weight_attr=paddle.ParamAttr(
                                 initializer=paddle.nn.initializer.Uniform(-stdv, stdv)))

    def forward(self, inputs):
        print('ResNet:')
        print('in', inputs.shape)
        y = self.conv(inputs)
        print(y.shape)
        y = self.pool2d_max(y)
        print('c1', y.shape)
        # bottleneck_block表示c2,c3,c4,c5
        for i, bottleneck_block in enumerate(self.bottleneck_block_list):
            y = bottleneck_block(y)
            # print('c2-c5{}'.format(i), y.shape)
        y = self.pool2d_avg(y)
        print('pool_avg', y.shape)
        y = paddle.reshape(y, [y.shape[0], -1])
        print('reshape', y.shape)
        y = self.out(y)
        print('out', y.shape)
        time.sleep(9999)
        return y

model = ResNet()
opt = paddle.optimizer.Momentum(learning_rate=0.001, momentum=0.9, parameters=model.parameters(), weight_decay=0.001)
train_pm(model, opt)