# -*- coding: utf-8 -*-

import time
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.metrics import mutual_info_score
import pickle
import os

def read_file(file_name):
    '''
    read data from given file name

    Parameters
    ----------
    file_name: str
                file path and name

    Returns:
    --------
    features: ndarray
                feature vector of instances
    labels: ndarray
                label vector of instances, label 0 indicates positive review while label 1 indicates negative review
    '''
    features, labels = [], []
    with open(file_name, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip(' \n').split(' ')
            temp = []
            for i in range(len(line) - 1):
                word_count = line[i].split(':')
                word, count = word_count[0], int(word_count[1])
                assert(type(word) == type('a'))
                temp.extend([word] * count)
            features.append(' '.join(temp))
            labels.append(1 if line[-1].split(':')[1] == 'positive' else 0)
    return features, labels

def get_top_MI(num, x, y):
    '''
    get top NUM words with highest mutual information

    Parameters
    ----------
    num: int
            top NUM words
    x: ndarray
            feature vector
    y: ndarray
            label vector

    Returns
    -------
    MI_sorted:  list
                index of top NUM words with highest mutual information, lower index in MI_sorted with higher mutual information
    MI: list
        MI[i] indicates mutual information score of feature i
    '''
    MI = []
    n = x.shape[1]
    for i in range(n):
        MI.append(mutual_info_score(x[:, i], y))
    MI_sorted = sorted(range(n), key = lambda i: MI[i])[-num :]
    MI_sorted.reverse()
    return MI_sorted, MI

def get_counts(count, index):
    return sum(count[:, index])

def get_data(src, tar, pivot_num, pivot_min_times):
    '''
    get data from source domain and target domain with given number and minimum appearance threshold of pivot features in both domains

    Parameters
    -----------
    src: string
        path of source domain
    tar: string
        path of target domain
    pivot_num: int
        the number of pivot features
    pivot_min_times: int
        the minimum appearance threshold of pivot features

    Returns
    -------
    x_non_pivot: ndarray
        non-pivot feature vectors of unlabeled data in both source and target domains
    x_pivot: ndarray
        pivot feature vectors of unlabeled data in both source and target domains
    D: int
        the number of non-pivot features
    '''
    prefix = '../dataset/Sentiment/'
    suffix1, suffix2, suffix3 = '/unlabeled.review', '/positive.review', '/negative.review'
    start_time = time.time()
    src_unl, src_unl_label = read_file(prefix + src + suffix1)
    src_pos, src_pos_label = read_file(prefix + src + suffix2)
    src_neg, src_neg_label = read_file(prefix + src + suffix3)
    tar_unl, tar_unl_label = read_file(prefix + tar + suffix1)
    # tar_pos, tar_pos_label = read_file(tar + '/positive.review')
    # tar_neg, tar_neg_label = read_file(tar + '/negative.review')

    read_file_time = time.time()
    print('read file ok --------------- time = %.3fs' % (read_file_time - start_time))

    # ---------- one solution for feature representation
    '''
    src_lab = src_pos + src_neg
    cv_src_lab = CountVectorizer(min_df = 10)
    x_src_lab = cv_src_lab.fit_transform(src_lab).toarray()

    lab_unl = src_unl + src_lab + tar_unl
    cv_lab_unl = CountVectorizer(min_df = 40)
    x_lab_unl = cv_lab_unl.fit_transform(lab_unl).toarray()

    src_data = src_pos + src_neg + src_unl
    cv_src = CountVectorizer(min_df = 20)
    x_src = cv_src.fit_transform(src_data).toarray()

    cv_tar_unl = CountVectorizer(min_df = 20)
    x_tar_unl = cv_tar_unl.fit_transform(tar_unl).toarray()
    '''

    # ----------- another solution for feature representation
    transformer = TfidfTransformer()

    src_lab = src_pos + src_neg
    cv_src_lab = CountVectorizer(min_df = 10)
    x_src_lab = transformer.fit_transform(cv_src_lab.fit_transform(src_lab)).toarray()

    lab_unl = src_unl + src_lab + tar_unl
    cv_lab_unl = CountVectorizer(min_df=40)
    x_lab_unl = transformer.fit_transform(cv_lab_unl.fit_transform(lab_unl)).toarray()

    src_data = src_pos + src_neg + src_unl
    cv_src = CountVectorizer(min_df=20)
    x_src = transformer.fit_transform(cv_src.fit_transform(src_data)).toarray()

    cv_tar_unl = CountVectorizer(min_df=20)
    x_tar_unl = transformer.fit_transform(cv_tar_unl.fit_transform(tar_unl)).toarray()


    cv_time = time.time()
    print('CountVectorize fit ok ------ time = %.3fs' % (cv_time - read_file_time))

    MI_sorted, MI = get_top_MI(5000, x_src_lab, src_pos_label + src_neg_label)

    get_mi_time = time.time()
    print('get MI ok ------------------ time = %.3fs' % (get_mi_time - cv_time))

    '''
    steps of choosing pivot features:
    1. get feature name from labeled instances in source domain
    2. get appearance time of the feature in source domain (labeled data + unlabeled data) and target domain (unlabeled data), respectively
    3. check if both appearance time is not less than pivot_min_times, if so, go to step 4; otherwise go to step 1
    4. store pivot feature name and index in the domain which contains both labeled data and unlabeled data in source domain and unlabeled data in target domain
    '''

    have, index = 0, 0
    pivot_name, pivot_index = [], []
    while have < pivot_num:
        name = cv_src_lab.get_feature_names()[MI_sorted[index]]
        src_count = get_counts(x_src, cv_src.get_feature_names().index(name)) if name in cv_src.get_feature_names() else 0
        tar_unl_count = get_counts(x_tar_unl, cv_tar_unl.get_feature_names().index(name)) if name in cv_tar_unl.get_feature_names() else 0

        if src_count >= pivot_min_times and tar_unl_count >= pivot_min_times:
            pivot_name.append(name)
            pivot_index.append(cv_lab_unl.get_feature_names().index(name))
            have += 1
        index += 1
        if index % 50 == 0:
            print('index = %d len(pivot_name) = %d len(pivot_index) = %d' % (index, len(pivot_name), len(pivot_index)))

    pivot_time = time.time()

    print('get pivot features ok ------ time = %.3fs' % (pivot_time - get_mi_time))
    pivot_path = './pivot/' + src + '2' + tar + '_pivot_' + str(pivot_num) + '_' + str(pivot_min_times)
    if not os.path.exists(os.path.dirname(pivot_path)):
        os.makedirs(os.path.dirname(pivot_path))
    with open(pivot_path, 'wb') as f:
        pickle.dump(pivot_index, f)

    x_pivot     = x_lab_unl[:, pivot_index]
    x_non_pivot = np.delete(x_lab_unl, pivot_index, 1)
    D           = x_non_pivot.shape[1]
    return x_non_pivot, x_pivot, D

def save_svd(src, tar, dim, W):
    '''
    save learned SVD weights for transformation from pivot features

    Parameters
    ----------
    src: string
        name of source domain
    tar: string
        name of target domain
    dim: int
        the dimensions of SVD decomposition
    W: ndarray
        weight vector
    '''
    path = './weight/' + src + '2' + tar + '_svd_' + str(dim) + '.npy'
    if not os.path.exists(os.path.dirname(path)):
        os.makedirs(os.path.dirname(path))

    np.save(path, W)

if __name__ == '__main__':
    base_path = '../dataset/Sentimentt/'
    src = 'books'
    tar = 'kitchen'
    get_data(base_path + src, base_path + tar, 50, 10)