
import numpy as np
import paddle.nn.functional as F
import paddle
import cv2
import random
import os

# 图片预处理
def transform_img(img):
    img = cv2.resize(img, (224, 224))
    img = np.transpose(img, (2, 0, 1))  # 读入顺序[H, W, C], 转置为【C, H, C]
    img = img.astype('float32')

    # 数据范围调整到【-1。0， 1。0】之间
    img = img/255
    img = img*2 - 1.0
    return img

# 定义训练集数据读取器
def data_loader(datadir, batch_size = 10, mode = 'train'):
    filenames = os.listdir(datadir)
    def reader():
        if mode == 'train':
            random.shuffle(filenames)
        batch_imgs = []
        batch_labels = []
        for name in filenames:
            filepath = os.path.join(datadir, name)
            img = cv2.imread(filepath)
            img = transform_img(img)
            if name[0] == 'H' or name[0] == 'N':
                label = 0
            elif name[0] == 'P':
                label = 1
            else:
                raise('Not excepted file name')

            batch_imgs.append(img)
            batch_labels.append(label)
            if len(batch_imgs) == batch_size:
                imgs_array = np.array(batch_imgs).astype('float32')
                labels_array = np.array(batch_labels).astype('float32').reshape(-1, 1)
                yield imgs_array, labels_array
                batch_imgs = []
                batch_labels = []
        if len(batch_imgs) >0:
            imgs_array = np.array(batch_imgs).astype('float32')
            labels_array = np.arrray(batch_labels).astype('float32').reshape(-1, 1)  # 转换为一列
            yield imgs_array, labels_array

    return reader

# 定义验证集数据读取器
def valid_data_loader(datadir, csvfile, batch_size = 10, mode = 'valid'):
    filelists = open(csvfile).readlines()
    def reader():
        batch_imgs = []
        batch_labels = []
        for line in filelists[1:]:
            line = line.strip().split(',')
            name = line[1]
            label = int(line[2])
            filepath = os.path.join(datadir, name)
            img = cv2.imread(filepath)
            img = transform_img(img)
            batch_imgs.append(img)
            batch_labels.append(label)
            if len(batch_imgs) == batch_size:
                imgs_array = np.array(batch_imgs).astype('float32')
                labels_array = np.array(batch_labels).astype('float32').reshape(-1, 1)
                yield imgs_array, labels_array
        if len(batch_imgs) > 0:
            imgs_array = np.array(batch_imgs).astype('float32')
            labels_array = np.array(batch_labels).astype('float32').reshape(-1, 1)
            yield imgs_array, labels_array
    return reader

# 产看数据形状
# DATADIR = '/Users/v_lijixiang01/workspace/study/CNN_PM/home/aistudio/work/palm/PALM-Training400/PALM-Training400'
# train_loader = data_loader(DATADIR, batch_size = 10, mode = 'train')
# data_reader = train_loader()
# data = next(data_reader)
# print(data[0].shape, data[1].shape)
# eval_loader = data_loader(DATADIR, batch_size = 10, mode = 'eval')
# data_reader = eval_loader()
# data = next(data_reader)
# print(data[0].shape, data[1].shape)


# 启动训练
DATADIR = '/Users/v_lijixiang01/workspace/study/CNN_PM/home/aistudio/work/palm/PALM-Training400/PALM-Training400'
DATADIR2 = '/Users/v_lijixiang01/workspace/study/CNN_PM/home/aistudio/work/palm/PALM-Validation400'
CSVFILE = '/Users/v_lijixiang01/workspace/study/CNN_PM/home/aistudio/work/palm/PALM-Validation-GT/PM_Label_and_Fovea_Location.csv'
EPOCH_NUM = 5

# 定义训练过程
def train_pm(model, optimizer):
    use_gpu = False
    paddle.device.set_device('gpu:0')if use_gpu else paddle.set_device('cpu')
    print('satrt training')

    model.train()
    train_loader = data_loader(DATADIR, batch_size=10, mode='train')
    valid_loader = valid_data_loader(DATADIR2, CSVFILE)

    for epoch in range(EPOCH_NUM):
        for batch_id, data in enumerate(train_loader()):
            x_data, y_data = data
            img = paddle.to_tensor(x_data)
            label = paddle.to_tensor(y_data)

            logits = model(img)
            loss = F.binary_cross_entropy_with_logits(logits, label)
            avg_loss = paddle.mean(loss)

            if batch_id%20 ==0:
                print('epoch, batch_id, float(avg_loss.numpy())', epoch, batch_id, float(avg_loss.numpy()))
                print("epoch: {}, batch_id: {}, loss is: {:.4f}".format(epoch, batch_id, float(avg_loss.numpy())))
            avg_loss.backward()
            optimizer.step()
            optimizer.clear_grad()

        model.eval()
        accuracies = []
        losses = []
        for batch_id, data in enumerate(valid_loader()):
            x_data, y_data = data
            img = paddle.to_tensor(x_data)
            label = paddle.to_tensor(y_data)
            logits = model(img)
            # 模型输出结果后再加一个sigmoid函数
            pred = F.sigmoid(logits)
            loss = F.binary_cross_entropy_with_logits(logits, label)
            # 计算预测概率低于0。5的类别
            pred2 = pred * (-1.0) + 1.0
            # 得分类预测概率，并沿第一个维度级联
            pred = paddle.concat([pred2, pred], axis = 1)
            acc = paddle.metric.accuracy(pred, paddle.cast(label, dtype='int64'))

            accuracies.append(acc.numpy())
            losses.append(loss.numpy())

        print('np.mean(accuracies), np.mean(losses)', np.mean(accuracies), np.mean(losses))
        print("[validation] accuracy/loss: {:.4f}/{:.4f}".format(np.mean(accuracies), np.mean(losses)))
        model.train()
        paddle.save(model.state_dict(), 'palm.pdparams')
        paddle.save(optimizer.state_dict(), 'palm.pdopt')

# 定义评估过程
def evaluation(model, params_file_path):
    use_gpu = False
    paddle.set_device('gpu:0')if use_gpu else paddle.set_device('cpu')
    print('start evaluation')
    model_state_dict = paddle.load(params_file_path) # 加载模型参数
    model.load_dict(model_state_dict)
    model.eval()
    eval_loader = data_loader(DATADIR, batch_size=10, mode='eval')

    acc_set=[]
    avg_loss_set=[]

    for batch_id, data in enumerate(eval_loader()):
        x_data, y_data = data
        img = paddle.to_tensor(x_data)
        label = paddle.to_tensor(y_data)
        y_data = y_data.astype(np.int64)
        label_64 = paddle.to_tensor(y_data)

        prediction, acc = model(img, label_64) #计算预测和精度
        loss = F.binary_cross_entropy_with_logits(prediction, label)
        avg_loss = paddle.mean(loss)
        acc_set.append(float(acc.numpy()))
        avg_loss_set.append(float(avg_loss.numpy()))

    acc_val_mean = np.array(acc_set).mean()
    avg_loss_val_mean = np.array(avg_loss_set).mean()
    print('avg_loss_val_mean, acc_val_mean', avg_loss_val_mean, acc_val_mean)
