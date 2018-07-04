from mxnet import ndarray as nd
from mxnet import gluon
import numpy as np

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


class DataLoader(object):
    """similiar to gluon.data.DataLoader, but might be faster.

    The main difference this data loader tries to read more exmaples each
    time. But the limits are 1) all examples in dataset have the same shape, 2)
    data transfomer needs to process multiple examples at each time
    """
    def __init__(self, dataset, batch_size, shuffle, transform=None):
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.transform = transform

    def __iter__(self):
        data = self.dataset[:]
        X = data[0]
        y = nd.array(data[1])
        n = X.shape[0]
        if self.shuffle:
            idx = np.arange(n)
            np.random.shuffle(idx)
            X = nd.array(X.asnumpy()[idx])
            y = nd.array(y.asnumpy()[idx])

        for i in range(n//self.batch_size):
            if self.transform is not None:
                yield self.transform(X[i*self.batch_size:(i+1)*self.batch_size],
                                     y[i*self.batch_size:(i+1)*self.batch_size])
            else:
                yield (X[i*self.batch_size:(i+1)*self.batch_size],
                       y[i*self.batch_size:(i+1)*self.batch_size])

    def __len__(self):
        return len(self.dataset)//self.batch_size


def load_data_fashion_mnist(batch_size, resize=None, root="~/.mxnet/datasets/fashion-mnist"):
    def transform_mnist(data, label):  # 处理数据格式
        if resize:
            n = data.shape[0]
            new_data = nd.zeros((n, resize, resize, data.shape[3]))
            for i in range(n):
                new_data[i] = image.imresize(data[i], resize, resize)
            data = new_data
        return nd.transpose(data.astype('float32'), (0, 3, 1, 2)) / 255, label.astype('float32')

    mnist_train = gluon.data.vision.FashionMNIST(root=root, train=True, transform=None)
    mnist_test = gluon.data.vision.FashionMNIST(root=root, train=False, transform=None)
    # Transform later to avoid memory explosion.
    train_data = DataLoader(mnist_train, batch_size, shuffle=True, transform=transform_mnist)
    test_data = DataLoader(mnist_test, batch_size, shuffle=False, transform=transform_mnist)
    return (train_data, test_data)
