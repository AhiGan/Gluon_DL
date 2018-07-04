from mxnet import nd
import utils
from mxnet import autograd
from mxnet import gluon


# import sys
# sys.path.append('/Users/lnn/Code/Github/Gluon_DL')

def dropout(X, drop_probablity):
    keep_probablity = 1 - drop_probablity
    # 断言检测变量合理性
    assert 0 <= keep_probablity <= 1

    # keep_probablity为0时丢弃所有值
    if keep_probablity == 0:
        return X.zeros_like()

    # 随机选择一部分该层的输出作为丢弃元素
    mask = nd.random.uniform(0, 1, X.shape, ctx=X.context) < keep_probablity

    # 拉伸使shape保持不变
    scale = 1 / keep_probablity
    return mask * X * scale


# A = nd.arange(20).reshape((5,4))
# print('dropout: 0.0, output: ', dropout(A, 0.0))
# print('dropout: 0.5, output: ', dropout(A, 0.5))
# print('dropout: 1.0, output: ', dropout(A, 1.0))

# 数据获取
batch_size = 256
train_data, test_data = utils.load_data_fashion_mnist(batch_size)

# 建立模型
num_inputs = 28 * 28
num_output = 10

num_hidden1 = 256
num_hidden2 = 256
weight_scale = .01

w1 = nd.random_normal(shape=(num_inputs, num_hidden1), scale=weight_scale)
b1 = nd.zeros(num_hidden1)

w2 = nd.random_normal(shape=(num_hidden1, num_hidden2), scale=weight_scale)
b2 = nd.zeros(num_hidden2)

w3 = nd.random_normal(shape=(num_hidden2, num_output), scale=weight_scale)
b3 = nd.zeros(num_output)

params = [w1, b1, w2, b2, w3, b3]
for param in params:
    param.attach_grad()

drop_prob1 = 0.2
drop_prob2 = 0.5


def net(X, is_training = False):
    X = X.reshape((-1, num_inputs))  # -1表示Numpy会根据剩下的维度计算出数组的另外一个shape属性值
    # 第一层全连接
    h1 = nd.relu(nd.dot(X, w1) + b1)
    # 在第一层全连接后添加丢弃层
    if is_training:  h1 = dropout(h1, drop_prob1)
    # 第二层全连接
    h2 = nd.relu(nd.dot(h1, w2) + b2)
    # 在第二层全连接后添加丢弃层
    if is_training:  h2 = dropout(h2, drop_prob2)
    return nd.dot(h2, w3) + b3

# # 不添加dropout层
# def net(X):
#     X = X.reshape((-1, num_inputs))  # -1表示Numpy会根据剩下的维度计算出数组的另外一个shape属性值
#     # 第一层全连接
#     h1 = nd.relu(nd.dot(X, w1) + b1)
#
#     # 第二层全连接
#     h2 = nd.relu(nd.dot(h1, w2) + b2)
#
#     return nd.dot(h2, w3) + b3

# 训练
softmax_cross_entropy = gluon.loss.SoftmaxCrossEntropyLoss()

learning_rate = 0.5
epoches = 5

for epoch in range(epoches):
    train_loss = 0.
    train_acc = 0.
    for data, label in train_data:
        with autograd.record():
            output = net(data, True)
            loss = softmax_cross_entropy(output, label)

        loss.backward()
        utils.SGD(params, learning_rate / batch_size)

        train_loss = train_loss + nd.mean(loss).asscalar()
        train_acc = train_acc + utils.accuracy(output, label)

    test_acc = utils.evaluate_accuracy(test_data, net)
    print("Epoch %d.Loss: %f, Train acc: %f, Test acc: %f " % (epoch, train_loss / len(train_data)
          , train_acc / len(train_data), test_acc))
