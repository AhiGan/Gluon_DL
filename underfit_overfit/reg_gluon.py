# _*_ coding: utf-8 _*_

from mxnet import ndarray as nd
from mxnet import autograd
from mxnet import gluon
import mxnet as mx
import matplotlib as mpl

mpl.rcParams['figure.dpi'] = 120
import matplotlib.pyplot as plt

num_train = 20
num_test = 100
num_inputs = 200  # 输入是200维的

true_w = nd.ones(shape=(num_inputs, 1)) * 0.01
true_b = 0.05

# 生成数据集
X = nd.random.normal(shape=(num_test + num_train, num_inputs))

y = nd.dot(X, true_w) + true_b
y += .01 * nd.random.normal(shape=y.shape)

X_train, X_test = X[:num_train, :], X[num_train:, :]
y_train, y_test = y[:num_train], y[num_train:]

batch_size = 1
dataset_train = gluon.data.ArrayDataset(X_train, y_train)
data_iter_train = gluon.data.DataLoader(dataset_train, batch_size, shuffle=True)
square_loss = gluon.loss.L2Loss()


# print('x:', x[:5])
# print('X:', X[:5])
# print('y:', y[:5])
def test(net, X, y):
    return square_loss(net(X), y).mean().asscalar()


def train(weight_decay):
    net = gluon.nn.Sequential()
    with net.name_scope():
        net.add(gluon.nn.Dense(1))
    # net.initialize()
    net.collect_params().initialize(mx.init.Normal(sigma=1))

    # 设置默认参数
    learning_rate = 0.005
    epochs = 10

    # 默认SGD和均方误差,此处增加了weight_decay项
    trainer = gluon.Trainer(net.collect_params(), 'sgd', {'learning_rate': learning_rate,
                                                          'wd': weight_decay})

    # 保存训练和测试损失
    train_loss = []
    test_loss = []
    for e in range(epochs):
        for data, label in data_iter_train:
            with autograd.record():
                output = net(data)
                loss = square_loss(output, label)

            loss.backward()
            trainer.step(batch_size)
        train_loss.append(test(net,X_train,y_train))
        test_loss.append(test(net,X_test,y_test))

    # 打印结果
    plt.plot(train_loss)
    plt.plot(test_loss)
    plt.legend(['train', 'test'])
    plt.show()
    return ('learned weight[:10]:', net[0].weight.data()[:,:10],
            'learned bias', net[0].bias.data())


# 正常模型
print(train(9))
