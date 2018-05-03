# _*_ coding: utf-8 _*_
# 当你的源代码中包含中文的时候，在保存源代码时，就需要务必指定保存为UTF-8编码
from mxnet import ndarray as nd
from mxnet import autograd as ag  # 这一行把autograd打错了一直导入包报错
from mxnet import gluon

# 生成数据
num_inputs = 2
num_examples = 1000

# 生成数据
num_inputs = 2
num_examples = 1000

true_w = [2, -3.4]
true_b = 4.2

X = nd.random_normal(shape=(num_examples, num_inputs))  # 二维数组间的逗号后需加空格
y = true_w[0] * X[:, 0] + true_w[1] * X[:, 1] + true_b
y += 0.01 * nd.random_normal(shape=y.shape)  # shape后的等号后不能加空格

# 读取数据
batch_size = 10
dataset = gluon.data.ArrayDataset(X, y)
data_iter = gluon.data.DataLoader(dataset, batch_size, shuffle=True)  # 这个函数自动按照batchsize生成打乱后的data批
for data, label in data_iter:
    print(data, label)
    break

# 模型:Sequential将所有层串起来，依次执行每一层，并将前一层的输出作为下一层的输入
net = gluon.nn.Sequential()  # 定义一个空的模型
net.add(gluon.nn.Dense(1))  # 增加一个Dense层，定于的参数是输出节点的个数

# 初始化模型参数
net.initialize()  # 随机初始化

# 损失函数：平方误差函数
square_loss = gluon.loss.L2Loss()

# 优化
trainer = gluon.Trainer(net.collect_params(), 'sgd', {'learning_rate': 0.1})

# 训练
epoches = 5
for e in range(epoches):
    total_loss = 0
    for data, label in data_iter:
        with ag.record():
            output = net(data)
            loss = square_loss(output, label)
        loss.backward()
        trainer.step(batch_size)
        total_loss += nd.sum(loss).asscalar()
    print("Epoach: %d, average loss: %f" % (e, total_loss/num_examples))

# 比较真实模型与学得模型
dense = net[0]
print(true_w, dense.weight.data())
print(true_b, dense.bias.data())








