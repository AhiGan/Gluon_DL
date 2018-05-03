from mxnet import gluon
from mxnet import ndarray as nd
from mxnet import autograd as ag
import utils


def transform(data, label):  # 处理数据格式
    return data.astype('float32')/255, label.astype('float32')


# 读取数据
mnist_train = gluon.data.vision.FashionMNIST(train=True, transform=transform)  # 这里直接使用data.vision来自动下载数据集
mnist_test = gluon.data.vision.FashionMNIST(train=False, transform=transform)

batch_size = 256
train_data = gluon.data.DataLoader(mnist_train, batch_size, shuffle=True)
test_data = gluon.data.DataLoader(mnist_test, batch_size, shuffle=False)  # 测试数据不需要将样本随机打乱

# 初始化模型参数
num_inputs = 784  # 图片像素为 28*28=784
num_outputs = 10  # 输出为10个类

w = nd.random_normal(shape=(num_inputs, num_outputs))
b = nd.random_normal(shape=num_outputs)
params = [w, b]
for param in params:  # 对模型参数附上剃度
    param.attach_grad()


def softmax(X):  # 通过Softmax函数将任意输入归一化称合法的概率值
    exp = nd.exp(X)  # 指数
    # 对行求和
    partition = exp.sum(axis=1, keepdims=True)  # keepdims=True 保持其二维特性
    return exp / partition


def net(X):
    return softmax(nd.dot(X.reshape((-1, num_inputs)), w) + b)  # reshape 第一项为-1表示自动填补


def cross_entropy(yhat, y):  # 交叉熵，因为此处yvec中只有一个1
    return - nd.pick(nd.log(yhat), y)  # 返回key为y对应的log值


# def accuracy(output, label):
#     return nd.mean(output.argmax(axis=1) == label).asscalar()
#
#
# def evaluate_accuracy(data_iterator, net):  # 模型在数据集上的精度
#     acc = 0.
#     for data, label in data_iterator:
#         output = net(data)
#         acc += accuracy(output, label)
#     return acc/len(data_iterator)
#
#
# def SGD(params, lr):  # 优化：随机剃度下降，lr-优化率
#     for param in params:
#         param[:] = param - lr * param.grad

# print(utils.evaluate_accuracy(test_data, net))

# 训练
epoches = 5
learning_rate = 0.1

for e in range(epoches):
    train_losses = 0.
    train_acc = 0.
    for data, label in train_data:
        with ag.record():
            output = net(data)
            loss = cross_entropy(output, label)
        loss.backward()
        utils.SGD(params, learning_rate/batch_size)  # 将剃度做平均使学习率对batch_size不敏感
        train_losses += nd.mean(loss).asscalar()
        train_acc += utils.accuracy(output, label)

    test_acc = utils.evaluate_accuracy(test_data, net)
    print("Epoach: %d, Loss: %f, Train Acc: %f, Test Acc: %f" %
          (e, train_losses/len(train_data), train_acc/len(train_data), test_acc))






