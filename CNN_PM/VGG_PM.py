import time

import paddle.nn
from paddle.nn import Conv2D, MaxPool2D, BatchNorm2D, Linear
from prepro_reader import *

class VGG(paddle.nn.Layer):
    def __init__(self):
        super(VGG, self).__init__()
        in_channels = [3, 64, 128, 256, 512, 512]

        self.conv1_1 = Conv2D(in_channels=in_channels[0], out_channels=in_channels[1], kernel_size=3, padding=1, stride=1)
        self.conv1_2 = Conv2D(in_channels=in_channels[1], out_channels=in_channels[1], kernel_size=3, padding=1, stride=1)

        self.conv2_1 = Conv2D(in_channels=in_channels[1], out_channels=in_channels[2], kernel_size=3, padding=1, stride=1)
        self.conv2_2 = Conv2D(in_channels=in_channels[2], out_channels=in_channels[2], kernel_size=3, padding=1, stride=1)

        self.conv3_1 = Conv2D(in_channels=in_channels[2], out_channels=in_channels[3], kernel_size=3, padding=1, stride=1)
        self.conv3_2 = Conv2D(in_channels=in_channels[3], out_channels=in_channels[3], kernel_size=3, padding=1, stride=1)
        self.conv3_3 = Conv2D(in_channels=in_channels[3], out_channels=in_channels[3], kernel_size=3, padding=1, stride=1)

        self.conv4_1 = Conv2D(in_channels=in_channels[3], out_channels=in_channels[4], kernel_size=3, padding=1, stride=1)
        self.conv4_2 = Conv2D(in_channels=in_channels[4], out_channels=in_channels[4], kernel_size=3, padding=1, stride=1)
        self.conv4_3 = Conv2D(in_channels=in_channels[4], out_channels=in_channels[4], kernel_size=3, padding=1, stride=1)

        self.conv5_1 = Conv2D(in_channels=in_channels[4], out_channels=in_channels[5], kernel_size=3, padding=1, stride=1)
        self.conv5_2 = Conv2D(in_channels=in_channels[5], out_channels=in_channels[5], kernel_size=3, padding=1, stride=1)
        self.conv5_3 = Conv2D(in_channels=in_channels[5], out_channels=in_channels[5], kernel_size=3, padding=1, stride=1)

        self.fc1 = paddle.nn.Sequential(paddle.nn.Linear(512*7*7, 4096), paddle.nn.ReLU())
        self.drop1_ratio = 0.5
        self.dropout1 = paddle.nn.Dropout(self.drop1_ratio, mode='upscale_in_train')
        self.fc2 = paddle.nn.Sequential(paddle.nn.Linear(4096, 4096), paddle.nn.ReLU())
        self.drop2_ratio = 0.5
        self.dropout2 = paddle.nn.Dropout(self.drop2_ratio, mode='upscale_in_train')

        self.fc3 = paddle.nn.Linear(4096, 1)
        self.relu = paddle.nn.ReLU()
        self.pool = MaxPool2D(stride=2, kernel_size=2)

    def forward(self, x):
        print(x.shape)
        x = self.relu(self.conv1_1(x))
        print(x.shape)
        x = self.relu(self.conv1_2(x))
        print(x.shape)
        x = self.pool(x)
        print(x.shape)

        x = self.relu(self.conv2_1(x))
        print(x.shape)
        x = self.relu(self.conv2_2(x))
        print(x.shape)
        x = self.pool(x)
        print(x.shape)

        x = self.relu(self.conv3_1(x))
        print(x.shape)
        x = self.relu(self.conv3_2(x))
        print(x.shape)
        x = self.relu(self.conv3_3(x))
        print(x.shape)
        x = self.pool(x)
        print(x.shape)

        x = self.relu(self.conv4_1(x))
        print(x.shape)
        x = self.relu(self.conv4_2(x))
        print(x.shape)
        x = self.relu(self.conv4_3(x))
        print(x.shape)
        x = self.pool(x)
        print(x.shape)

        x = self.relu(self.conv5_1(x))
        print(x.shape)
        x = self.relu(self.conv5_2(x))
        print(x.shape)
        x = self.relu(self.conv5_3(x))
        print(x.shape)
        x = self.pool(x)
        print(x.shape)

        x = paddle.flatten(x, 1, -1)
        print(x.shape)
        x = self.dropout1(self.relu(self.fc1(x)))
        print(x.shape)
        x = self.dropout2(self.relu(self.fc2(x)))
        print(x.shape)
        x = self.fc3(x)
        print(x.shape)
        time.sleep(9999)

        return x

# 创建模型
model = VGG()
opt = paddle.optimizer.Momentum(learning_rate=0.001, momentum=0.9, parameters=model.parameters())
train_pm(model, opt)