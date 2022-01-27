from paddle.nn import Conv2D, MaxPool2D, Linear
from prepro_reader import *


# 定义LeNet网络
class LeNet(paddle.nn.Layer):
    def __init__(self, num_classes = 1):
        super(LeNet, self).__init__()
        self.conv1 = Conv2D(in_channels=3, out_channels=6, kernel_size=5)
        self.max_pool1 = MaxPool2D(kernel_size=2, stride=2)
        self.conv2 = Conv2D(in_channels=6, out_channels=16, kernel_size=5)
        self.max_pool2 = MaxPool2D(kernel_size=2, stride=2)

        self.conv3 = Conv2D(in_channels=16, out_channels=120, kernel_size=4)

        self.fc1 = Linear(in_features=300000, out_features=64)
        self.fc2 = Linear(in_features=64, out_features=num_classes)

    def forward(self, x, label=None):
        x = self.conv1(x)
        x = F.sigmoid(x)
        x = self.max_pool1(x)
        x = self.conv2(x)
        x = F.sigmoid(x)
        x = self.max_pool2(x)
        x = self.conv3(x)
        x = F.sigmoid(x)
        x = paddle.reshape(x, [x.shape[0], -1])
        x = self.fc1(x)
        x = F.sigmoid(x)
        x = self.fc2(x)
        if label is not None:
            acc = paddle.metric.accuracy(input=x, label=label)
            return x, acc
        else:
            return x

model = LeNet()
opt = paddle.optimizer.Momentum(learning_rate=0.001, momentum=0.9, parameters=model.parameters())
train_pm(model, optimizer=opt)
evaluation(model, params_file_path="palm.pdparams")