# -*- coding: utf-8 -*-

import random
import numpy as np
from sklearn import svm
import time
import warnings
warnings.filterwarnings("ignore")

def train(X, Y, P, test_x, base_learner):
    clf = svm.SVC(probability=True)
    clf.fit(X, Y, P)
    test_predict = clf.predict_proba(test_x)[:, 1]
    train_predict = clf.predict_proba(X)[:, 1]
    return train_predict, test_predict

def TraAdaBoost(src_x, src_y, val_x, val_y, test_x, test_y, epoch = 100, base_learner = 'svm'):
    # initialize weights
    n, m, r = src_x.shape[0], val_x.shape[0], test_x.shape[0]
    src_w = np.ones(n) / n
    tar_w = np.ones(m) / m

    X = np.concatenate((src_x, val_x), axis = 0)
    Y = np.concatenate((src_y, val_y), axis = 0)
    W = np.concatenate((src_w, tar_w), axis = 0)

    beta0 = 1.0 / (1.0 + np.sqrt(2.0 * np.log(n) / epoch)) # see detail from source code: https://www.cse.ust.hk/TL/
    beta = np.zeros(epoch)
    test_output = np.zeros([test_x.shape[0], epoch])
    for i in range(epoch):
        epoch_start_time = time.time()

        P = W / np.sum(W)
        train_predict, test_predict = train(X, Y, P, test_x, base_learner)
        test_output[:, i] = test_predict

        assert (train_predict.shape[0] == Y.shape[0])

        epsilon = np.sum(np.multiply(W[n:n+m], np.fabs(train_predict[n:n+m] - Y[n:n+m]))) / np.sum(W[n:n+m])
        if epsilon > 0.5:
            epsilon = 0.5
        if epsilon == 0.5 or epsilon == 0:
            epoch = i
            break

        beta[i] = epsilon / (1 - epsilon)
        for j in range(n):
            W[j] *= np.power(beta0, -np.abs(train_predict[j] - Y[j]))
        for j in range(n, n + m):
            W[j] *= np.power(beta[i], np.abs(train_predict[j] - Y[j]))

        right1 = np.sum(np.sign(train_predict[:n] - 0.5) == np.sign(Y[:n] - 0.5))
        right2 = np.sum(np.sign(train_predict[n:n+m] - 0.5) == np.sign(Y[n:n+m] - 0.5))
        right3 = np.sum(np.sign(test_predict - 0.5) == np.sign(test_y - 0.5))
        if i % 10 == 0:
            print('epcoh = %d src_acc = %.4f val_acc = %.4f train_acc = %.4f test_acc = %.4f time = %.3fs' % (i, right1/n, right2/m, (right1+right2)/(n+m), right3 / r, time.time() - epoch_start_time))


    test_predict = np.zeros(r)
    st = int(np.ceil(epoch / 2))
    # epoch = max(epoch, st + 10)

    # threshold = np.prod([beta[i] ** -0.5 for i in range(st, epoch)])
    # print('beta = %s' % (' '.join([str(i) for i in beta[:epoch]])))
    # print('threshold = %.4f\n' % (threshold))
    for i in range(r):
        # left = np.prod([beta[i] ** (-test_output[i]) for i in range(st, epoch)])
        # right = np.prod([beta[i] ** -0.5 for i in range(st, epoch)])
        s = np.sum(np.multiply(0.5 - test_output[i, st:epoch], np.log(beta[st:epoch])))
        # print(str(s), end = ' ')
        if s >= 0:
        # if left > right:
            test_predict[i] = 1
    '''
    print('\ntest_predict = %s' % (' '.join([str(i) for i in test_predict[:epoch]])))
    print('test_output = %s' % (' '.join([str(i) for i in test_output[0, st:epoch]])))
    print('test_y = %s' % (' '.join([str(i) for i in test_y])))
    print('epoch = %d' % (epoch))
    '''

    acc1 = np.sum(test_predict == test_y) / r
    clf = svm.SVC(probability=True)
    clf.fit(X, Y)
    test_predict = clf.predict_proba(test_x)[:, 1]
    acc2 = np.sum(np.sign(test_predict - 0.5) == np.sign(test_y - 0.5)) / r

    return acc1, acc2


if __name__ == '__main__':
    # read source data and target data
    src = []
    domain = 3
    for i in range(1, domain + 1):
        with open('../dataset/landmine/domain' + str(i) + '.txt', 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip(' \n').split(' ')
                line = [float(x) for x in line]
                src.append(line)

    tar = []
    with open('../dataset/landmine/domain' + str(20) + '.txt', 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip(' \n').split(' ')
            line = [float(i) for i in line]
            tar.append(line)

    src, tar = np.array(src), np.array(tar)

    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    src_tar = scaler.fit_transform(np.concatenate((src[:, :-2], tar[:, :-2]), axis = 0))
    src[:, :-2] = src_tar[:src.shape[0], :]
    tar[:, :-2] = src_tar[src.shape[0]:, :]
    print('read source data and target data ok !')

    # split target data into validation set and test data
    val_index = random.sample(range(len(tar)), int(0.05 * len(tar)))
    tar_val = tar[val_index]
    tar_test = []
    for i in range(0, len(tar)):
        if i not in val_index:
            tar_test.append(tar[i])
    tar_test = np.array(tar_test)

    n, m, r = src.shape[0], tar_val.shape[0], tar_test.shape[0]
    src_x, src_y = src[:,:-2], src[:,-1]
    tar_val_x, tar_val_y = tar_val[:,:-2], tar_val[:,-1]
    tar_test_x, tar_test_y = tar_test[:,:-2], tar_test[:,-1]

    print('src_x.shape = {} src_y.shape = {} tar_val_x.shape = {} tar_val_y.shape = {} tar_test_x.shape = {} tar_test_y.shape = {}'.format( \
            src_x.shape, src_y.shape, tar_val_x.shape, tar_val_y.shape, tar_test_x.shape, tar_test_y.shape))

    for round in range(10):
        acc1, acc2 = TraAdaBoost(src_x, src_y, tar_val_x, tar_val_y, tar_test_x, tar_test_y)
        print("TraAdaBoost_acc = %.6f SVM_acc = %.6f" % (acc1, acc2))


