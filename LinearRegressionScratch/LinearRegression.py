# _*_ coding: utf-8 _*_
# 当你的源代码中包含中文的时候，在保存源代码时，就需要务必指定保存为UTF-8编码
from mxnet import ndarray as nd
from mxnet import autograd as ag  # 这一行把autograd打错了一直导入包报错
from matplotlib import pyplot as plt  # Python绘图包
import random

# 生成数据
num_inputs = 2
num_examples = 1000

true_w = [2, -3.4]
true_b = 4.2

X = nd.random_normal(shape=(num_examples, num_inputs))  # 二维数组间的逗号后需加空格
y = true_w[0] * X[:, 0] + true_w[1] * X[:, 1] + true_b
y += 0.01 * nd.random_normal(shape=y.shape)  # shape后的等号后不能加空格

# print(X[0], y[0])
# print(X)
# print(y)

# 散点图
# plt.scatter(X[:, 1].asnumpy(), y.asnumpy())
# plt.show()

# 读取数据，批大小设为10
batch_size = 10


# 迭代器返回随机打乱后的数据
def data_iter():
    idx = list(range(num_examples))
    random.shuffle(idx)  # 随机打乱序列
    for i in range(0, num_examples, batch_size):
        j = nd.array(idx[i:min(i + batch_size, num_examples)])
        yield nd.take(X, j), nd.take(y, j)  # yield是个迭代器，相当于在此返回take获得的内容，然后从下一句继续执行


# for data, label in data_iter():
#     print(data, label)
#     break

w = nd.random_normal(shape=(num_inputs, 1))
b = nd.zeros((1,))
params = [w, b]

# 剃度
for param in params:
    param.attach_grad()


def net(X):  # 计算预测值
    return nd.dot(X, w) + b


def square_loss(yhat, y):  # 平方损失
    # 将y变形成yhat的形状，避免矩阵形状的自动转换
    return (yhat - y.reshape(yhat.shape)) ** 2


def SGD(params, lr):  # 优化：随机剃度下降，lr-优化率
    for param in params:
        param[:] = param - lr * param.grad


def real_fn(X):  # 模型函数
    return true_w[0] * X[:, 0] + true_w[1] * X[:, 1] + true_b


def plot(losses, X, sample_size=100):  # 绘制损失随训练次数降低的折线图，以及预测值和真实值的散点图
    xs = list(range(len(losses)))
    f, (fg1, fg2) = plt.subplots(1, 2)
    fg1.set_title('Loss during training')
    fg1.plot(xs, losses, '-r')
    fg2.set_title('Estimated vs real function')
    fg2.plot(X[:sample_size, 1].asnumpy(), net(X[:sample_size, :]).asnumpy(), 'or', label='Estimated')
    fg2.plot(X[:sample_size, 1].asnumpy(), real_fn(X[:sample_size, :]).asnumpy(), '*g', label='real')
    fg2.legend()
    plt.show()


epoches = 5  # 迭代执行五次
learning_rate = .001
niter = 0
losses = []
moving_loss = 0
smoothing_constant = .01
# 训练

for e in range(epoches):
    total_loss = 0

    for data, label in data_iter():
        with ag.record():
            output = net(data)
            loss = square_loss(output, label)
        loss.backward()
        SGD(params, learning_rate)
        total_loss += nd.sum(loss).asscalar()

        # 记录每读取一批数据点后，损失的移动平均值的变化
        niter += 1
        curr_loss = nd.mean(loss).asscalar()
        moving_loss = (1 - smoothing_constant) * moving_loss + smoothing_constant * curr_loss


        # correct the bias from the moving averages
        est_loss = moving_loss/(1-(1-smoothing_constant)**niter)

        if(niter+1)%100 == 0:
            losses.append(est_loss)
            print("Epoch %s, batch %s, Moving avg of losses: %s, Average loss: %f"%(e, niter, est_loss,total_loss/num_examples))
            plot(losses, X)