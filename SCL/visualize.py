# -*- coding: utf-8 -*-

domain = ['books', 'kitchen', 'dvd', 'electronics']
# result = ['baseline', 'ori', '50_30', '100_50', '150_100']
result = ['baseline', 'baseline_v2', 'SCL', 'SCL_MI', '50_30', '50_30_v2', '100_50', '150_100']
NUM = 8

X, Y = [], []

for path in result:
    file = './results/results_' + path + '/'
    x, y = [], []
    for src in domain:
        acc = []
        name = []
        for tar in domain:
            if src == tar:
                continue
            name.append(tar)
            file_name = file + src + '2' + tar + '.txt'
            with open(file_name, 'r', encoding = 'utf-8') as f:
                for line in f:
                    acc.append(float(line.strip(' \n').split(' ')[-2].split('=')[1]))

        x.append(name)
        y.append(acc)
    X.append(x)
    Y.append(y)

import matplotlib.pyplot as plt
plt.figure()

color = ['g', 'b', 'r', 'y', 'k', 'm', 'pink', 'teal']
for i in range(NUM):
    for j in range(4):
        ax = plt.subplot(1, 4, j + 1)
        ax.set_title(domain[j])
        plt.plot(X[i][j], Y[i][j], color[i], label = result[i])
        plt.legend()

plt.show()