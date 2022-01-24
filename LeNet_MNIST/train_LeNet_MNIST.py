
from paddle.vision.transforms import ToTensor
from paddle.vision.datasets import MNIST
from show_layers import *

def train(model, opt, train_loader, valid_loader):
    use_gpu = False
    paddle.device.set_device('gpu:0') if use_gpu else paddle.device.set_device('cpu')
    print('start training')
    model.train()
    for epoch in range(EPOCH_NUM):
        for batch_id, data in enumerate(train_loader):
            img = data[0]
            label = data[1]

            logits =model(img)  # 计算模型输出
            loss_func = paddle.nn.CrossEntropyLoss(reduction='none')    # 计算损失函数
            loss = loss_func(logits, label)
            avg_loss = paddle.mean(loss)

            if batch_id % 2000 ==0:
                print("epoch:{},batch_id:{}, loss is :{:.4f}".format(epoch, batch_id, float(avg_loss.numpy())))
            avg_loss.backward()
            opt.step()
            opt.clear_grad()

        model.eval()
        accuracies = []
        losses = []
        for batch_id, data in enumerate(valid_loader):
            img = data[0]
            label = data[1]

            logits = model(img)
            pred = F.softmax(logits)
            loss_func = paddle.nn.CrossEntropyLoss(reduction="none")
            loss = loss_func(logits, label)
            acc = paddle.metric.accuracy(pred, label)
            accuracies.append(acc.numpy())
            losses.append(loss.numpy())
        print("[validation] accuracy/loss: {:.4f}/{:.4f}".format(np.mean(accuracies), np.mean(losses)))

        model.train()
    paddle.save(model.state_dict(), 'mnist.pdparams')

model = LeNet(num_classes = 10)
EPOCH_NUM = 5
opt = paddle.optimizer.Momentum(learning_rate=0.001, momentum=0.9, parameters=model.parameters())
train_loader = paddle.io.DataLoader(MNIST(mode = 'train', transform=ToTensor()), batch_size=10, shuffle=True)
valid_loader = paddle.io.DataLoader(MNIST(mode='test', transform=ToTensor()), batch_size=10)

train(model, opt, train_loader, valid_loader)


