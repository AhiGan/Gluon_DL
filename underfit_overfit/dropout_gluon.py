from mxnet import nd
import utils
from mxnet import autograd
from mxnet import gluon
from mxnet.gluon import nn

# 数据获取
batch_size = 256
train_data, test_data = utils.load_data_fashion_mnist(batch_size)


# 建立模型
net = nn.Sequential()
drop_prob1= 0.2
drop_prob2= 0.5

flag = True

with net.name_scope():

    net.add(nn.Flatten())
    # 第一层全连接
    net.add(nn.Dense(256,activation="relu"))
    # 在第一层全连接后添加丢弃层
    if flag:
        net.add(nn.Dropout(drop_prob1))
    # 第二层全连接层
    net.add(nn.Dense(256,activation="relu"))
    # 在第二层全连接后添加丢弃层
    if flag:
        net.add(nn.Dropout(drop_prob2))
    net.add(nn.Dense(10))

net.initialize()

softmax_cross_entropy = gluon.loss.SoftmaxCrossEntropyLoss()
trainer = gluon.Trainer(net.collect_params(),
                        'sgd', {'learning_rate': 0.5})

epoches = 5

for epoch in range(epoches):
    train_loss = 0.
    train_acc = 0.
    for data, label in train_data:
        with autograd.record():
            output = net(data)
            loss = softmax_cross_entropy(output, label)

        loss.backward()
        trainer.step(batch_size)

        train_loss = train_loss + nd.mean(loss).asscalar()
        train_acc = train_acc + utils.accuracy(output, label)

    flag = False
    test_acc = utils.evaluate_accuracy(test_data, net)
    flag = True

    print("Epoch %d.Loss: %f, Train acc: %f, Test acc: %f " % (epoch, train_loss / len(train_data)
          , train_acc / len(train_data), test_acc))