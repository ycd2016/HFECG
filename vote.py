import os
import pandas as pd
import numpy as np
import catboost as cb
import sklearn as sk
from sklearn.metrics import f1_score
from sklearn.model_selection import KFold
from config import config
from data_process import name2index
top = pd.read_csv('deep.txt',
                  header=None,
                  sep='\t').drop([0],
                                 axis=1).fillna(-1)
top.ix[top[2] == 'FEMALE', 2] = 0
top.ix[top[2] == 'MALE', 2] = 1
subB = np.concatenate((np.array(top), np.array(pd.read_csv('feature.csv'))),
                      axis=1)
for cur in range(5):
    oneout = np.zeros((subB.shape[0], 34))
    for i in range(34):
        cls = cb.CatBoostClassifier(task_type='GPU', devices='0')
        cls.load_model('cbm/' + str(i) + '_' + str(cur) + '.cbm')
        oneout[:, i] = cls.predict_proba(subB)[:, 1]
    np.savetxt(str(cur) + '.txt', oneout, '%.09f', '\t')
for cur in range(5):
    oneout = np.zeros((subB.shape[0], 34))
    for i in range(34):
        cls = cb.CatBoostClassifier(task_type='GPU', devices='0')
        cls.load_model('cbm_/' + str(i) + '_' + str(cur) + '.cbm')
        oneout[:, i] = cls.predict_proba(subB)[:, 1]
    np.savetxt(str(cur) + '_.txt', oneout, '%.09f', '\t')
a = np.loadtxt('0.txt')
s = np.loadtxt('1.txt')
d = np.loadtxt('2.txt')
f = np.loadtxt('3.txt')
g = np.loadtxt('4.txt')
h = (a + s + d + f + g) / 5.
np.savetxt('h.txt', h, '%.09f', '\t')
lim = [0.41, 0.38, 0.47, 0.49, 0.49, 0.52, 0.42, 0.41, 0.34, 0.29, 0.43, 0.53,
       0.37, 0.35, 0.38, 0.35, 0.42, 0.53, 0.45, 0.49, 0.46, 0.60, 0.33, 0.44,
       0.33, 0.33, 0.29, 0.44, 0.35, 0.30, 0.24, 0.36, 0.45, 0.39]
for i in range(34):
    h[:, i] = h[:, i] > lim[i]
a = np.loadtxt('0_.txt')
s = np.loadtxt('1_.txt')
d = np.loadtxt('2_.txt')
f = np.loadtxt('3_.txt')
g = np.loadtxt('4_.txt')
hh = (a + s + d + f + g) / 5.
np.savetxt('hh.txt', hh, '%.09f', '\t')
lim = [0.36, 0.44, 0.10, 0.47, 0.55, 0.55, 0.41, 0.39, 0.39, 0.23, 0.57, 0.36,
       0.48, 0.36, 0.35, 0.40, 0.42, 0.41, 0.46, 0.39, 0.48, 0.42, 0.37, 0.39,
       0.17, 0.37, 0.27, 0.26, 0.25, 0.50, 0.39, 0.48, 0.39, 0.29]
for i in range(34):
    hh[:, i] = hh[:, i] > lim[i]
ecoc = np.array(pd.read_csv('ecoc.csv', header=None))
ecout = np.zeros_like(hh)
minhmd = np.zeros((h.shape[0], ))
h = np.concatenate((h, hh), axis=1)
for i in range(h.shape[0]):
    minhmd[i] = ecoc.shape[1] + 1
    for j in range(ecoc.shape[0]):
        minhmd[i] = min(np.sum(h[i] != ecoc[j]), minhmd[i])
    for j in range(ecoc.shape[0]):
        if (np.sum(h[i] != ecoc[j]) == minhmd[i]):
            ecout[i, j] = 1
    if (np.sum(ecout[i]) > 5):
        ecout[i] = hh[i]
print(np.sum(ecout, axis=0).astype('int'))
np.savetxt('ecout.txt', ecout, '%d', '\t')
cur = 0
name2idx = name2index(config.arrythmia)
idx2name = {idx: name for name, idx in name2idx.items()}
sub_file = './result.txt'
fout = open(sub_file, 'w', encoding='utf-8')
for line in open(config.test_label, encoding='utf-8'):
    fout.write(line.strip('\n'))
    output = ecout[cur]
    ixs = [i for i, out in enumerate(output) if out > 0.5]
    for i in ixs:
        fout.write("\t" + idx2name[i])
    fout.write('\n')
    cur = cur + 1
fout.close()
