# -*- coding: utf-8 -*-

from sklearn.preprocessing import StandardScaler
import numpy as np
import os
import random

val_ratio = 0.05
labeled_ratio = 0.1

def save_data(file_name, x, y):
    with open(file_name, 'w', encoding = 'utf-8') as f:
        for i in range(len(x)):
            line = x[i] + [y[i]]
            line_ = [str(i) for i in line]
            f.write(' '.join(line_) + '\n')

def data_split(num, x, y, round):
    assert (len(x) == len(y))

    N = len(set(y))
    samples = [[] for _ in range(N)]

    for i in range(len(x)):
        assert(y[i] >= 0)
        samples[y[i]].append(x[i] + [y[i]])

    for i in range(N):
        random.shuffle(samples[i])

    x_tar_val, y_tar_val = [], []
    x_tar_train, y_tar_train = [], []

    for i in range(N):
        n = len(samples[i])
        tar_val_num = int(n * val_ratio)

        x_tar_val.extend(samples[i][: tar_val_num])
        x_tar_train.extend(samples[i][tar_val_num: ])

    random.shuffle(x_tar_val)
    random.shuffle(x_tar_train)

    for i in range(len(x_tar_val)):
        y_tar_val.append(x_tar_val[i][-1])
        x_tar_val[i] = x_tar_val[i][:-1]

    for i in range(len(x_tar_train)):
        y_tar_train.append(x_tar_train[i][-1])
        x_tar_train[i] = x_tar_train[i][:-1]

    labeled_num = int(len(x_tar_train) * labeled_ratio)
    x_tar_lab, y_tar_lab = x_tar_train[:labeled_num], y_tar_train[:labeled_num]
    x_tar_unl, y_tar_unl = x_tar_train[labeled_num:], y_tar_train[labeled_num:]
    x_tar_test, y_tar_test = x_tar_unl, y_tar_unl

    path = '../domain%d/%d/' % (num, round)
    if not os.path.exists(path):
        os.makedirs(path)

    save_data(path + 'tar_test.txt', x_tar_test, y_tar_test)
    save_data(path + 'tar_val.txt', x_tar_val, y_tar_val)
    save_data(path + 'tar_lab.txt', x_tar_lab, y_tar_lab)
    save_data(path + 'tar_unl.txt', x_tar_unl, y_tar_unl)

def normalize(X, Y):
    X, Y = np.array(X), np.array(Y)
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    return X.tolist(), Y.tolist()

def get_all_data(num):
    X, Y = [], []
    with open(('domain%d.txt' % num), 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip(' \n').split(' ')
            for i in range(len(line) - 1):
                line[i] = float(line[i])
            if len(line[-1]) > 1:
                line[-1] = (ord(line[-1][0]) - ord('0')) * 10 + ord(line[-1][1]) - ord('0')
            else:
                line[-1] = ord(line[-1]) - ord('0')
            X.append(line[:-1])
            Y.append(line[-1])
    return X, Y

if __name__ == '__main__':
    source_x, source_y = [], []
    for i in range(1, 6):
        tx, ty = get_all_data(i)
        source_x.extend(tx)
        source_y.extend(ty)
    save_data('../source.txt', source_x, source_y)
    source_x, source_y = normalize(source_x, source_y)
    save_data('../source_norm.txt', source_x, source_y)

    for i in range(6, 30):
        x, y = get_all_data(i)
        x, y = normalize(x, y)
        for round in range(30):
            data_split(i, x, y, round)