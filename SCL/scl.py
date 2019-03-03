# -*- coding: utf-8 -*-

import data_helper
import numpy as np
import time
import pickle
import os
from sklearn import linear_model
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.linear_model import LogisticRegression

def train(src, tar, pivot_num, pivot_min_times, dim):
    get_data_start = time.time()
    x_non_pivot, x_pivot, D = data_helper.get_data(src, tar, pivot_num, pivot_min_times)
    pivot_matrix = np.zeros((pivot_num, D))
    sgd_start = time.time()
    print('get data ok ---------------- time = %.3fs' % (sgd_start - get_data_start))

    # SVD decomposition based on pivot features and non-pivot features
    for i in range(pivot_num):
        iteration_start = time.time()
        # clf = linear_model.SGDClassifier(loss = 'modified_huber', max_iter = 20, tol = 1e-3)
        clf = linear_model.SGDRegressor(loss = 'huber', max_iter = 20, tol = 1e-3)
        clf.fit(x_non_pivot, x_pivot[:, i])
        pivot_matrix[i] = clf.coef_
        # pivot_matrix[i] = clf.coef_[:, -1]
        if i > 0 and i % 20 == 0:
            print('SGD iteration i = %d per iteration time = %.3fs' % (i, time.time() - iteration_start))
    pivot_matrix = pivot_matrix.transpose()

    sgd_end = time.time()
    print('SGD optimization ok -------- time = %.3fs' % (sgd_end - sgd_start))

    svd = TruncatedSVD(n_components = dim)
    weight = svd.fit_transform(pivot_matrix)
    data_helper.save_svd(src, tar, dim, weight)

def test_baseline(src, tar):
    prefix = '../dataset/Sentiment/'
    suffix = ['/positive.review', '/negative.review', '/unlabeled.review']
    src_pos, src_pos_label = data_helper.read_file(prefix + src + suffix[0])
    src_neg, src_neg_label = data_helper.read_file(prefix + src + suffix[1])
    tar_pos, tar_pos_label = data_helper.read_file(prefix + tar + suffix[0])
    tar_neg, tar_neg_label = data_helper.read_file(prefix + tar + suffix[1])

    transformer = TfidfTransformer()
    cv_src = CountVectorizer(min_df=20)
    # x_src = cv_src.fit_transform(src_pos + src_neg).toarray()
    x_src = transformer.fit_transform(cv_src.fit_transform(src_pos + src_neg)).toarray()

    y_src = src_pos_label + src_neg_label

    lr = LogisticRegression(solver='lbfgs', C=C)
    lr.fit(x_src, y_src)

    # x_tar = cv_src.transform(tar_pos + tar_neg).toarray()
    x_tar = transformer.fit_transform(cv_src.transform(tar_pos + tar_neg)).toarray()
    y_tar = tar_pos_label + tar_neg_label

    acc = lr.score(x_tar, y_tar)
    return acc

def test(src, tar, pivot_num, pivot_min_times, dim, C):
    weight_path = './weight/' + src + '2' + tar + '_svd_' + str(dim) + '.npy'
    W = np.load(weight_path)

    pivot_path = './pivot/' + src + '2' + tar + '_pivot_' + str(pivot_num) + '_' + str(pivot_min_times)
    with open(pivot_path, 'rb') as f:
        pivot = pickle.load(f)

    prefix = '../dataset/Sentiment/'
    suffix = ['/positive.review', '/negative.review', '/unlabeled.review']
    src_pos, src_pos_label = data_helper.read_file(prefix + src + suffix[0])
    src_neg, src_neg_label = data_helper.read_file(prefix + src + suffix[1])
    src_unl, src_unl_label = data_helper.read_file(prefix + src + suffix[2])
    tar_pos, tar_pos_label = data_helper.read_file(prefix + tar + suffix[0])
    tar_neg, tar_neg_label = data_helper.read_file(prefix + tar + suffix[1])
    tar_unl, tar_unl_label = data_helper.read_file(prefix + tar + suffix[2])

    # transformer_src = TfidfTransformer()
    cv_src = CountVectorizer(min_df = 20)
    x_src = cv_src.fit_transform(src_pos + src_neg).toarray()
    # x_src = transformer_src.fit_transform(cv_src.fit_transform(src_pos + src_neg)).toarray()

    cv_lab_unl = CountVectorizer(min_df = 40)
    x_lab_unl = cv_lab_unl.fit_transform(src_unl + src_pos + src_neg + tar_unl).toarray()
    # transformer_lab_unl = TfidfTransformer()
    # x_lab_unl = transformer_lab_unl.fit_transform(cv_lab_unl.fit_transform(src_unl + src_pos + src_neg + tar_unl)).toarray()

    # x_src_transform = transformer_lab_unl.transform(cv_lab_unl.transform(src_pos + src_neg)).toarray()
    x_src_transform = cv_lab_unl.transform(src_pos + src_neg).toarray()
    x_src_transform = np.delete(x_src_transform, pivot, 1)
    x_src_transform = x_src_transform.dot(W)

    x_src = np.concatenate((x_src, x_src_transform), axis = 1)
    y_src = src_pos_label + src_neg_label

    lr = LogisticRegression(solver = 'lbfgs', C = C)
    lr.fit(x_src, y_src)

    # x_tar_transform = transformer_lab_unl.transform(cv_lab_unl.transform(tar_pos + tar_neg)).toarray()
    x_tar_transform = cv_lab_unl.transform(tar_pos + tar_neg).toarray()
    x_tar_transform = np.delete(x_tar_transform, pivot, 1)
    x_tar_transform = x_tar_transform.dot(W)

    # x_tar = transformer_src.transform(cv_src.transform(tar_pos + tar_neg)).toarray()
    x_tar = cv_src.transform(tar_pos + tar_neg).toarray()
    x_tar = np.concatenate((x_tar, x_tar_transform), axis = 1)
    y_tar = tar_pos_label + tar_neg_label

    acc = lr.score(x_tar, y_tar)
    return acc

if __name__ == '__main__':
    pivot_num = 50
    pivot_min_times = 10
    dim = 30
    C = 0.1

    domain = ['books', 'kitchen', 'dvd', 'electronics']
    for src in domain:
        for tar in domain:
            if src == tar:
                continue

            start_time = time.time()

            train(src, tar, pivot_num, pivot_min_times, dim)
            print('training ok ---------------- time = %.3fs' % (time.time() - start_time))

            acc = test_baseline(src, tar)
            acc = test(src, tar, pivot_num, pivot_min_times, dim, C)
            end_time = time.time()

            path = './results/results_' + str(pivot_num) + '_' + str(dim) + '/'
            if not os.path.exists(path):
                os.makedirs(path)
            with open(path + ('%s2%s.txt' % (src, tar)), 'w', encoding='utf-8') as f:
                info = ('%s--->%s pivot_num=%d pivot_min_times=%d acc=%s time=%.3f' % (
                src, tar, pivot_num, pivot_min_times, str(acc), end_time - start_time))
                f.write(info + '\n')

            print(info + '\n')