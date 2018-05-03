from mxnet import ndarray as nd


def accuracy(output, label):
    return nd.mean(output.argmax(axis=1) == label).asscalar()


def evaluate_accuracy(data_iterator, net):  # 模型在数据集上的精度
    acc = 0.
    for data, label in data_iterator:
        output = net(data)
        acc += accuracy(output, label)
    return acc/len(data_iterator)


def SGD(params, lr):  # 优化：随机剃度下降，lr-优化率
    for param in params:
        param[:] = param - lr * param.grad