import matplotlib as mpl

mpl.rcParams['figure.dpi'] = 120
import matplotlib.pyplot as plt
from mxnet import gluon
from mxnet import autograd as ag
from mxnet import ndarray as nd
import housePricePrediction


def train(net, X_train, y_train, X_test, y_test, epochs,
          verbose_epoch, learning_rate, weight_decay):
    train_loss = []
    if y_test is not None:
        test_loss = []
    batch_size = 100
    dataset_train = gluon.data.ArrayDataset(X_train, y_train)
    data_iter_train = gluon.data.DataLoader(dataset_train, batch_size, shuffle=True)
    trainer = gluon.Trainer(net.collect_params(), 'adam', {'learning_rate': learning_rate, 'wd': weight_decay})
    net.collect_params().initialize(force_reinit=True)  # ??
    for epoch in range(epochs):
        for data, label in data_iter_train:
            with ag.record():
                output = net(data)
                loss = housePricePrediction.square_loss(output, label)
            loss.backward()
            trainer.step(batch_size)

            cur_train_loss = housePricePrediction.get_rmse_log(net, X_train, y_train)
        if epoch > verbose_epoch:
            print("Epoch %d, train loss: %f " % (epoch, cur_train_loss))

        train_loss.append(cur_train_loss)

        if X_test is not None:
            cur_test_loss = housePricePrediction.get_rmse_log(net, X_test, y_test)
            test_loss.append(cur_test_loss)

    plt.plot(train_loss)
    plt.legend('train')

    if X_test is not None:
        plt.plot(test_loss)
        plt.legend(['train', 'test'])

    plt.show()

    if X_test is not None:
        return cur_train_loss, cur_test_loss
    else:
        return cur_train_loss


def k_fold_cross_valid(k, epochs, verbose_epoch, X_train, y_train, learning_rate, weight_decay):
    assert k > 1
    fold_size = X_train.shape[0] // k  # python中//是除法取整
    train_loss_sum = 0.0
    test_loss_sum = 0.0
    for test_i in range(k):
        # 第test_i折的测试数据
        X_val_test = X_train[test_i * fold_size:(test_i + 1) * fold_size, :]
        y_val_test = y_train[test_i * fold_size:(test_i + 1) * fold_size]

        val_train_defined = False
        for i in range(k):  # 第test_i折的训练数据
            if i != test_i:
                X_cur_fold = X_train[i * fold_size:(i + 1) * fold_size, :]
                y_cur_fold = y_train[i * fold_size:(i + 1) * fold_size]
                if not val_train_defined:
                    X_val_train = X_cur_fold
                    y_val_train = y_cur_fold
                else:
                    X_val_train = nd.concat(X_val_train, X_cur_fold, dim=0)
                    y_val_train = nd.concat(y_cur_fold, y_cur_fold, dim=0)

        net = housePricePrediction.get_net()
        train_loss, test_loss = train(net, X_val_train, y_val_train, X_val_test, y_val_test, epochs,
                                      verbose_epoch, learning_rate, weight_decay)
        train_loss_sum += train_loss
        print("Test Loss: %f " % test_loss)  # 第test_i折的测试误差
        test_loss_sum += test_loss

    return train_loss_sum / k, test_loss_sum / k


k = 5  # 5折交叉验证
epochs = 100
verbose_epoch = 95
learning_rate = 5
weight_decay = 0.0

train_loss, test_loss = k_fold_cross_valid(k, epochs, verbose_epoch, X_train=housePricePrediction.X_train
                                           , y_train=housePricePrediction.y_train, learning_rate=learning_rate,
                                           weight_decay=weight_decay)

print("%d-fold validation: Avg train loss: %f, Avg test loss: %f " % (k, train_loss, test_loss))
