
from mxnet import autograd
from mxnet import gluon
from mxnet import image
from mxnet import init
from mxnet import nd
from mxnet.gluon import nn
from mxnet.gluon.data import vision
import numpy as np
import pandas as pd
import datetime
import sys
sys.path.append('..')
import utils

train_dir = 'train'
test_dir = 'test'
batch_size = 128

data_dir = 'data'
label_file = 'trainLabels.csv'
input_dir = 'train_valid_test'
valid_ratio = 0.1

# ============================================================= #
def transform_train(data, label):
    im = data.astype('float32') / 255
    auglist = image.CreateAugmenter(data_shape=(3, 32, 32), resize=0,
                        rand_crop=True, rand_resize=True, rand_mirror=True,
                        mean=np.array([0.4914, 0.4822, 0.4465]),
                        std=np.array([0.2023, 0.1994, 0.2010]),
                        brightness=0, contrast=0,
                        saturation=0, hue=0,
                        pca_noise=0, rand_gray=0, inter_method=2)
    for aug in auglist:
        im = aug(im)
    # 将数据格式从"高*宽*通道"改为"通道*高*宽"。
    im = nd.transpose(im, (2,0,1))
    return (im, nd.array([label]).asscalar().astype('float32'))

# 测试时，无需对图像做标准化以外的增强数据处理。
def transform_test(data, label):
    im = data.astype('float32') / 255
    auglist = image.CreateAugmenter(data_shape=(3, 32, 32),
                        mean=np.array([0.4914, 0.4822, 0.4465]),
                        std=np.array([0.2023, 0.1994, 0.2010]))
    for aug in auglist:
        im = aug(im)
    im = nd.transpose(im, (2,0,1))
    return (im, nd.array([label]).asscalar().astype('float32'))


input_str = data_dir + '/' + input_dir + '/'

# 读取原始图像文件。flag=1说明输入图像有三个通道（彩色）。
train_ds = vision.ImageFolderDataset(input_str + 'train', flag=1,
                                     transform=transform_train)
valid_ds = vision.ImageFolderDataset(input_str + 'valid', flag=1,
                                     transform=transform_test)
train_valid_ds = vision.ImageFolderDataset(input_str + 'train_valid',
                                           flag=1, transform=transform_train)
test_ds = vision.ImageFolderDataset(input_str + 'test', flag=1,
                                     transform=transform_test)

loader = gluon.data.DataLoader
train_data = loader(train_ds, batch_size, shuffle=True, last_batch='keep')
valid_data = loader(valid_ds, batch_size, shuffle=True, last_batch='keep')
train_valid_data = loader(train_valid_ds, batch_size, shuffle=True, last_batch='keep')
test_data = loader(test_ds, batch_size, shuffle=False, last_batch='keep')

# 交叉熵损失函数。
softmax_cross_entropy = gluon.loss.SoftmaxCrossEntropyLoss()


print("Data Loaded.")

# ================================================================ #

class Residual(nn.HybridBlock):
    def __init__(self, channels, same_shape=True, **kwargs):
        super(Residual, self).__init__(**kwargs)
        self.same_shape = same_shape
        with self.name_scope():
            strides = 1 if same_shape else 2
            self.bn1 = nn.BatchNorm()
            self.conv1 = nn.Conv2D(channels//4, kernel_size=1, use_bias=False)
            self.bn2 = nn.BatchNorm()
            self.conv2 = nn.Conv2D(channels//4, kernel_size=3, padding=1,
                                  strides=strides, use_bias=False)
            self.bn3 = nn.BatchNorm()
            self.conv3 = nn.Conv2D(channels, kernel_size=1, use_bias=False)
            self.bn4 = nn.BatchNorm()
            if not same_shape:
                self.conv4 = nn.Conv2D(channels, kernel_size=1,
                                      strides=strides)

    def hybrid_forward(self, F, x):
        out = self.conv1(F.relu(self.bn1(x)))
        out = self.conv2(F.relu(self.bn2(out)))
        out = self.conv3(F.relu(self.bn3(out)))
        out = self.bn4(out)
        if not self.same_shape:
            x = self.conv4(x)
        return out + x


class ResNet(nn.HybridBlock):
    def __init__(self, num_classes, verbose=False, **kwargs):
        super(ResNet, self).__init__(**kwargs)
        self.verbose = verbose
        with self.name_scope():
            net = self.net = nn.HybridSequential()
            # block 1
            net.add(nn.Conv2D(channels=32, kernel_size=3, strides=1, padding=1, use_bias=False))
            # block 2
            for _ in range(27):
                net.add(Residual(channels=32))
            # block 3
            net.add(Residual(channels=64, same_shape=False))
            for _ in range(26):
                net.add(Residual(channels=64))
            # block 4
            net.add(Residual(channels=128, same_shape=False))
            for _ in range(26):
                net.add(Residual(channels=128))
            # block 5
            net.add(nn.BatchNorm())
            net.add(nn.Activation('relu'))
            net.add(nn.AvgPool2D(pool_size=8))
            net.add(nn.Flatten())
            net.add(nn.Dense(num_classes))

    def hybrid_forward(self, F, x):
        out = x
        for i, b in enumerate(self.net):
            out = b(out)
            if self.verbose:
                print('Block %d output: %s'%(i+1, out.shape))
        return out


def get_net(ctx):
    num_outputs = 10
    net = ResNet(num_outputs)
    net.initialize(ctx=ctx, init=init.Xavier())
    return net

# ================================================================ #

def train(net, train_data, valid_data, num_epochs, lr, wd, ctx, lr_periods, lr_decay, f):
    trainer = gluon.Trainer(
        net.collect_params(), 'sgd', {'learning_rate': lr, 'momentum': 0.9, 'wd': wd})

    prev_time = datetime.datetime.now()
    
    sum_time = 0
    for epoch in range(num_epochs):
        train_loss = 0.0
        train_acc = 0.0
        if epoch > 0 and epoch in lr_periods:
            trainer.set_learning_rate(trainer.learning_rate * lr_decay)
        
        for data, label in train_data:
            label = label.as_in_context(ctx)
            with autograd.record():
                output = net(data.as_in_context(ctx))
                loss = softmax_cross_entropy(output, label)
            loss.backward()
            trainer.step(batch_size)
            train_loss += nd.mean(loss).asscalar()
            train_acc += utils.accuracy(output, label)
        
        cur_time = datetime.datetime.now()
        h, remainder = divmod((cur_time - prev_time).seconds, 3600)
        m, s = divmod(remainder, 60)
        time_str = "Time %02d:%02d:%02d" % (h, m, s)

        # Estimate remaing time
        sum_time += (cur_time - prev_time).seconds
        rs = (sum_time / (epoch + 1)) * (num_epochs - epoch - 1)
        h, remainder = divmod(rs, 3600)
        m, s = divmod(remainder, 60)
        print("Estimate: %02d:%02d:%02d" % (h, m, s))

        if valid_data is not None:
            valid_acc = utils.evaluate_accuracy(valid_data, net, ctx)
            epoch_str = ("Epoch %d. Loss: %f, Train acc %f, Valid acc %f, "
                         % (epoch, train_loss / len(train_data),
                            train_acc / len(train_data), valid_acc))
            f.write("%d,%f,%f,%f\n" % (epoch, train_loss / len(train_data), train_acc / len(train_data), valid_acc))
        else:
            epoch_str = ("Epoch %d. Loss: %f, Train acc %f, "
                         % (epoch, train_loss / len(train_data),
                            train_acc / len(train_data)))
            f.write("%d,%f,%f\n" % (epoch, train_loss / len(train_data), train_acc / len(train_data)))

        prev_time = cur_time
        print(epoch_str + time_str + ', lr ' + str(trainer.learning_rate))

        filename = "data/params/resnet.params"
        net.save_params(filename)


ctx = utils.try_gpu()
num_epochs = 240
learning_rate = 0.1
weight_decay = 1e-4
lr_periods = [160, 200, 220]
lr_decay = 0.1

# net = get_net(ctx)
# net.hybridize()
# train(net, train_data, valid_data, num_epochs, learning_rate,
#       weight_decay, ctx, lr_periods, lr_decay)


net = get_net(ctx)
net.hybridize()
with open("out_log.txt", "w+") as f:
    train(net, train_valid_data, None, num_epochs, learning_rate, 
      weight_decay, ctx, lr_periods, lr_decay, f)

preds = []
for data, label in test_data:
    output = net(data.as_in_context(ctx))
    preds.extend(output.argmax(axis=1).astype(int).asnumpy())

sorted_ids = list(range(1, len(test_ds) + 1))
sorted_ids.sort(key = lambda x:str(x))

df = pd.DataFrame({'id': sorted_ids, 'label': preds})
df['label'] = df['label'].apply(lambda x: train_valid_ds.synsets[x])
df.to_csv('submission.csv', index=False)
