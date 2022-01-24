import paddle
import numpy as np
from paddle.nn import Conv2D, MaxPool2D, Linear

# 组网
import paddle.nn.functional as F

# 定义LeNet网络结构
class LeNet(paddle.nn.Layer):
    def __init__(self, num_classes = 1):

        # 函数是用于调用父类(超类)的一个方法。
        super(LeNet, self).__init__()
        # 创建卷积和池化层
        # 创建第1个卷积层
        self.conv1 = Conv2D(in_channels=1, out_channels=6, kernel_size=5)
        self.max_pool1 = MaxPool2D(kernel_size=2, stride =2)
        # 尺寸的逻辑，池化层不改变通道数
        # 创建第2个卷积层
        self.conv2 = Conv2D(in_channels=6, out_channels=16, kernel_size=5)
        self.max_pool2 = MaxPool2D(kernel_size=2, stride = 2)
        # 创建第3个卷积层
        self.conv3 = Conv2D(in_channels=16, out_channels=120, kernel_size=4)
        # 尺寸逻辑：输入层数将数据拉平[B,C,H,W] -> [B,C*H*W]
        # 输入size是28*28，经三次卷积池化之后，C*H*W 为120
        self.fc1 = Linear(in_features=120, out_features=64)
        # 创建全连接层，第一个全连接层的输出神经元个数为64， 第二个全连接层输出神经元个数为分类标签的类别数
        self.fc2 = Linear(in_features=64, out_features=num_classes)
    # 网络的前向计算过程
    def forward(self, x):
        x = self.conv1(x)
        # 每个卷积层后面加一个sigmoid函数，后面接一个2*2池化层
        x = F.sigmoid(x)
        x = self.max_pool1(x)
        x = F.sigmoid(x)
        x = self.conv2(x)
        x = self.max_pool2(x)
        x = self.conv3(x)
        # 尺寸的逻辑：将数据输入层数将数据拉平[B,C,H,W] -> [B,C*H*W]
        x = paddle.reshape(x, [x.shape[0], -1])
        x = self.fc1(x)
        x = F.sigmoid(x)
        x = self.fc2(x)
        return x

# 输入数据形状是 [N, 1, H, W]
# 这里用np.random创建一个随机数组作为输入数据
x = np.random.randn(*[3,1,28,28]) # *的意思是列表中的元素，都是独立参数
x = x.astype('float32')

# 创建LeNet类的实例，指定模型名称和分类的类别数目
model = LeNet(num_classes=10)
# 通过调用LeNet从基类继承的sublayers()函数，
# 查看LeNet中所包含的子层

x = paddle.to_tensor(x)


for item in model.sublayers():
    # item是LeNet类中的一个子层
    # 查看经过子层之后的输出数据形状

    # print('\nx.shape:', x.shape )
    # print('item:', item)
    # print('[x.shape[0],-1]:', [x.shape[0],-1])
    # print('item.parameters():', item.parameters()[0].shape)
    try:
        # item()返回的是一个浮点型数据
        x = item(x)
    except:
        x = paddle.reshape(x,[x.shape[0],-1])
        x = item(x)
    if len(item.parameters()) == 2:
    #     # 查看卷积和全连接层的数据和参数的形状，
    #     # 其中item.parameters()[0]是权重参数w，item.parameters()[1]是偏置参数bi
        print(item.full_name(), x.shape, item.parameters()[0].shape, item.parameters()[1].shape)
    else:
    #     # 池化层没有参数
        print(item.full_name(), x.shape)

