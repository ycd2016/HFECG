import os, torch
import numpy as np
from config import config
np.random.seed(0)


def name2index(path):
    list_name = []
    for line in open(path, encoding='utf-8'):
        list_name.append(line.strip())
    name2indx = {name: i for i, name in enumerate(list_name)}
    return name2indx


def split_data(file2idx, val_ratio=0.1):
    data = set(os.listdir(config.train_dir))
    val = set()
    idx2file = [[] for _ in range(config.num_classes)]
    for file, list_idx in file2idx.items():
        for idx in list_idx:
            idx2file[idx].append(file)
    for item in idx2file:
        print(len(item), item)
        num = int(len(item) * val_ratio)
        val = val.union(item[:num])
    train = data.difference(val)
    return list(train), list(val)


def file2index(path, name2idx):
    file2index = dict()
    for line in open(path, encoding='utf-8'):
        arr = line.strip().split('\t')
        id = arr[0]
        labels = [name2idx[name] for name in arr[3:]]
        file2index[id] = labels
    return file2index


def count_labels(data, file2idx):
    cc = [0] * config.num_classes
    for fp in data:
        for i in file2idx[fp]:
            cc[i] += 1
    return np.array(cc)


def train(name2idx, idx2name):
    file2idx = file2index(config.train_label, name2idx)
    train, val = split_data(file2idx)
    wc = count_labels(train, file2idx)
    print(wc)
    dd = {
        'train': train,
        'val': val,
        "idx2name": idx2name,
        'file2idx': file2idx,
        'wc': wc
    }
    torch.save(dd, config.train_data)


if __name__ == '__main__':
    name2idx = name2index(config.arrythmia)
    idx2name = {idx: name for name, idx in name2idx.items()}
    train(name2idx, idx2name)
