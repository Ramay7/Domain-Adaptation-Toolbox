# -*- coding: utf-8 -*-

import numpy as np
import pickle
import random
import os
import time

# Note: 需要将样本的标记表示成{-1, 1}的形式

class SVM:
    def __init__(self, X, Y, weight = 'none', max_iter = 100, kernel_opt = 'rbf', gamma = 'none', C = 1.0, toler = 0.001, min_alpha_change = 0.00001):
        self.train_x = X    # 训练特征
        self.train_y = Y    # 训练标签
        self.n = np.shape(X)[0]  # 训练样本的个数
        self.weight = weight if weight != 'none' else np.ones((self.n, 1))

        self.max_iter = max_iter
        self.C = C                                  # 惩罚参数
        self.toler = toler                          # 迭代的终止条件之一
        self.min_alpha_change = min_alpha_change    # 最小系数改变量

        self.alphas = np.mat(np.zeros((self.n, 1)))  # 拉格朗日乘子
        self.b = 0
        self.error_tmp = np.mat(np.zeros((self.n, 2)))  # 保存E的缓存

        self.kernel_opt = kernel_opt  # 选用的核函数及其参数
        self.gamma = gamma if gamma != 'none' else 1.0 / np.shape(X)[1]

        self.kernel_mat = self.calc_kernel()  # 核函数的输出

    def cal_kernel_value(self, train_x_i):
        '''样本之间的核函数的值
        input:  train_x_i(mat):第i个训练样本的特征向量
        output: kernel_value(mat):样本之间的核函数的值

        '''
        kernel_value = np.mat(np.zeros((self.n, 1)))
        if self.kernel_opt == 'rbf':  # rbf核函数
            for i in range(self.n):
                diff = self.train_x[i, :] - train_x_i
                # print('diff.shape = {} diff.T.shape = {} kernel_value[i].shape = {}'.format(diff.shape, diff.T.shape, kernel_value[i].shape))
                kernel_value[i] = np.exp(np.sum(diff * diff.T) / (-2.0 * self.gamma**2))
        else:  # 不使用核函数
            kernel_value = self.train_x * train_x_i.T
        return kernel_value

    def calc_kernel(self):
        '''计算核函数矩阵
        input:
        output: kernel_matrix(mat):样本的核函数的值
        '''
        kernel_matrix = np.mat(np.zeros((self.n, self.n)))  # 初始化样本之间的核函数值
        for i in range(self.n):
            kernel_matrix[:, i] = self.cal_kernel_value(self.train_x[i, :])
        return kernel_matrix

    def cal_error(self, index):
        '''误差值的计算
        input:  index(int):选择出的变量
        output: error_k(float):误差值
        '''
        # print('self.alphas.shape = {} self.train_y.shape = {} self.kernel_mat[:, index].shape = {}'.format(self.alphas.shape, self.train_y.shape, self.kernel_mat[:, index].shape))
        # print('(self.alphas, self.train_y).T.shape = {}'.format(np.multiply(self.alphas, self.train_y).T.shape))
        output = float(np.multiply(self.alphas, self.train_y).T * self.kernel_mat[:, index] + self.b)
        error = (output - float(self.train_y[index])) * self.weight[index]
        return error

    def update_error_tmp(self, index):
        '''重新计算误差值
        input:  alpha_k(int):选择出的变量
        output: 对应误差值
        '''
        error = self.cal_error(index)
        self.error_tmp[index] = [1, error]

    def select_second_sample_j(self, alpha_i, error_i):
        '''选择第二个样本
        input:  alpha_i(int):选择出的第一个变量
                error_i(float):E_i
        output: alpha_j(int):选择出的第二个变量
                error_j(float):E_j
        '''
        # 标记为已被优化
        self.error_tmp[alpha_i] = [1, error_i]
        candidateAlphaList = np.nonzero(self.error_tmp[:, 0].A)[0]

        max_error, alpha_j, error_j = 0, 0, 0

        if len(candidateAlphaList) > 1:
            for index in candidateAlphaList:
                if index == alpha_i:
                    continue
                error = self.cal_error(index)
                if abs(error - error_i) > max_error:
                    max_error= abs(error - error_i)
                    alpha_j = index
                    error_j = error
        else:  # 随机选择
            alpha_j = alpha_i
            while alpha_j == alpha_i:
                alpha_j = int(np.random.uniform(0, self.n))
            error_j = self.cal_error(alpha_j)

        return alpha_j, error_j

    def choose_and_update(self, alpha_i):
        '''判断和选择两个alpha进行更新
        input: alpha_i(int):选择出的第一个变量
        '''
        error_i = self.cal_error(alpha_i)  # 计算第一个样本的E_i

        # 判断选择出的第一个变量是否违反了KKT条件
        if (self.train_y[alpha_i] * error_i < -self.toler) and (self.alphas[alpha_i] < self.C) or \
            (self.train_y[alpha_i] * error_i > self.toler) and (self.alphas[alpha_i] > 0):

            # 1、选择第二个变量
            alpha_j, error_j = self.select_second_sample_j(alpha_i, error_i)
            alpha_i_old = self.alphas[alpha_i].copy()
            alpha_j_old = self.alphas[alpha_j].copy()

            # 2、计算上下界
            if self.train_y[alpha_i] != self.train_y[alpha_j]:
                L = max(0, self.alphas[alpha_j] - self.alphas[alpha_i])
                H = min(self.C, self.C + self.alphas[alpha_j] - self.alphas[alpha_i])
            else:
                L = max(0, self.alphas[alpha_j] + self.alphas[alpha_i] - self.C)
                H = min(self.C, self.alphas[alpha_j] + self.alphas[alpha_i])
            if L == H:
                # print('L === H, return 0')
                return 0

            # 3、计算eta
            eta = 2.0 * self.kernel_mat[alpha_i, alpha_j] - self.kernel_mat[alpha_i, alpha_i] \
                  - self.kernel_mat[alpha_j, alpha_j]
            if eta >= 0:
                # print('eta >= 0, return 0')
                return 0

            # 4、更新alpha_j
            self.alphas[alpha_j] -= self.train_y[alpha_j] * (error_i - error_j) / eta

            # 5、确定最终的alpha_j
            if self.alphas[alpha_j] > H:
                self.alphas[alpha_j] = H
            if self.alphas[alpha_j] < L:
                self.alphas[alpha_j] = L

            # 6、判断是否结束
            if abs(alpha_j_old - self.alphas[alpha_j]) < self.min_alpha_change:
                self.update_error_tmp(alpha_j)
                # print('update alpha_j within self.min_alpha_change, return 0')
                return 0

            # 7、更新alpha_i
            self.alphas[alpha_i] += self.train_y[alpha_i] * self.train_y[alpha_j] * (alpha_j_old - self.alphas[alpha_j])

            # 8、更新b
            b1 = self.b - error_i - self.train_y[alpha_i] * (self.alphas[alpha_i] - alpha_i_old) * self.kernel_mat[alpha_i, alpha_i] \
                 - self.train_y[alpha_j] * (self.alphas[alpha_j] - alpha_j_old) * self.kernel_mat[alpha_i, alpha_j]
            b2 = self.b - error_j - self.train_y[alpha_i] * (self.alphas[alpha_i] - alpha_i_old) * self.kernel_mat[alpha_i, alpha_j] \
                 - self.train_y[alpha_j] * (self.alphas[alpha_j] - alpha_j_old) * self.kernel_mat[alpha_j, alpha_j]

            if (0 < self.alphas[alpha_i]) and (self.alphas[alpha_i] < self.C):
                self.b = b1
            elif (0 < self.alphas[alpha_j]) and (self.alphas[alpha_j] < self.C):
                self.b = b2
            else:
                self.b = (b1 + b2) / 2.0

            # 9、更新error
            self.update_error_tmp(alpha_j)
            self.update_error_tmp(alpha_i)
            return 1
        else:
            return 0

    def smo(self):
        entireSet = True
        alpha_pairs_changed = 0
        iteration = 0

        # print('train samples = %d dimensions = %d' % (self.n, self.train_x.shape[1]))
        while (iteration < self.max_iter) and ((alpha_pairs_changed > 0) or entireSet):
            alpha_pairs_changed = 0
            if entireSet:   # 对所有的样本
                for index in range(self.n):
                    alpha_pairs_changed += self.choose_and_update(index)
            else:           # 非边界样本
                bound_samples = []
                for i in range(self.n):
                    if self.alphas[i, 0] > 0 and self.alphas[i, 0] < self.C:
                        bound_samples.append(i)
                for x in bound_samples:
                    alpha_pairs_changed += self.choose_and_update(x)

            iteration += 1

            # 在所有样本和非边界样本之间交替
            if entireSet:
                entireSet = False
            elif alpha_pairs_changed == 0:
                entireSet = True

            # print('iteration = %d alpha_pairs_changed = %d' % (iteration, alpha_pairs_changed))

    def predict(self, test_sample_x):
        '''利用SVM模型对每一个样本进行预测
        input:  test_sample_x(mat):样本
        output: predict(float):对样本的预测
        '''
        m = test_sample_x.shape[0]
        predict = np.ones((m, 1))
        for i in range(m):
            # 1、计算核函数矩阵
            kernel_value = self.cal_kernel_value(test_sample_x[i])
            # 2、计算预测值
            predict[i] = kernel_value.T * np.multiply(self.train_y, self.alphas) + self.b
        return predict

    def evaluate(self, test_x, test_y):
        '''计算预测的准确性
        input:  test_x(mat):测试的特征
                test_y(mat):测试的标签
        output: accuracy(float):预测的准确性
        '''
        n_samples = np.shape(test_x)[0]  # 样本的个数
        predict = self.predict(test_x)
        correct = 0.0
        for i in range(n_samples):
            # predict = self.predict(test_x[i, :])
            if np.sign(predict[i]) == np.sign(test_y[i]):
                correct += 1
        acc = correct / n_samples
        return acc

    def save(self, path, name):
        '''保存SVM模型
        input:  model_file(string):SVM模型需要保存到的文件
        '''
        if not os.path.exists(path):
            os.makedirs(path)

        with open(path + name, 'w', encoding = 'utf-8') as f:
            pickle.dump(self, f)


def load_data_libsvm(data_file):
    '''导入训练数据
    input:  data_file(string):训练数据所在文件
    output: data(mat):训练样本的特征
            label(mat):训练样本的标签
    '''
    data = []
    label = []
    f = open(data_file)
    for line in f.readlines():
        lines = line.strip().split(' ')

        # 提取得出label
        label.append(float(lines[0]))
        # 提取出特征，并将其放入到矩阵中
        index = 0
        tmp = []
        for i in range(1, len(lines)):
            li = lines[i].strip().split(":")
            if int(li[0]) - 1 == index:
                tmp.append(float(li[1]))
            else:
                while (int(li[0]) - 1 > index):
                    tmp.append(0)
                    index += 1
                tmp.append(float(li[1]))
            index += 1
        while len(tmp) < 13:
            tmp.append(0)
        data.append(tmp)
    f.close()
    return np.mat(data), np.mat(label).T

if __name__ == '__main__':
    start_time = time.time()
    X = []
    domain = 2
    for i in range(1, domain + 1):
        with open('../dataset/landmine/domain' + str(i) + '.txt', 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip(' \n').split(' ')
                line = [float(x) for x in line]
                X.append(line)
    X = np.array(X)

    test_index = random.sample(range(len(X)), int(0.2 * len(X)))
    test = X[test_index]
    train = []
    for i in range(0, len(X)):
        if i not in test_index:
            train.append(X[i])
    train = np.array(train)

    train_x, train_y = train[:, :-2], (train[:,-1] - 0.5) * 2
    test_x, test_y = test[:, :-2], (test[:,-1] - 0.5) * 2

    train_y = np.reshape(train_y, (len(train_y), 1))
    test_y = np.reshape(test_y, (len(test_y), 1))

    print('train_x.shape = {} train_y.shape = {} test_x.shape = {} test_y.shape = {}'.format(train_x.shape, train_y.shape, test_x.shape, test_y.shape))

    clf = SVM(train_x, train_y)
    clf.smo()

    path = './svm_model/'
    name = 'svm_model'
    # clf.save(path, name)
    # if not os.path.exists(path):
    #    os.makedirs(path)
    # with open(path + name, 'w', encoding='utf-8') as f:
    #     pickle.dump(clf, f)

    acc_train = clf.evaluate(train_x, train_y)
    acc_test = clf.evaluate(test_x, test_y)
    print('acc_train = %.6f acc_test = %.6f time = %.3f' % (acc_train, acc_test, time.time() - start_time))
    # acc_train = 0.9990942028985508 acc_test = 0.9311594202898551
    # acc_train = 1.0 acc_test = 0.9239130434782609 // 0.938406
    from sklearn import svm
    svm.SVC()
    '''
    from sklearn import svm
    start_time = time.time()
    clf = svm.SVC()
    clf.fit(train_x, train_y)
    n, m = train_x.shape[0], test_x.shape[0]
    test_predict = clf.predict(test_x).reshape((m, 1))
    train_predict = clf.predict(train_x).reshape((n, 1))
    acc_train = np.sum(train_predict == train_y) / n
    acc_test = np.sum(test_predict == test_y) / m
    print('sklearn: acc_train = %.6f acc_test = %.6f time = %.3f' % (acc_train, acc_test, time.time() - start_time))
    '''