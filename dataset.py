import pywt, os, copy
import torch
import numpy as np
import pandas as pd
from config import config
from torch.utils.data import Dataset
from sklearn.preprocessing import scale
from scipy import signal


def resample(sig, target_point_num=None):
    sig = signal.resample(sig, target_point_num) if target_point_num else sig
    return sig


def scaling(X, sigma=0.05):
    scalingFactor = np.random.normal(loc=1.0,
                                     scale=sigma,
                                     size=(1, X.shape[1]))
    myNoise = np.matmul(np.ones((X.shape[0], 1)), scalingFactor)
    return X * myNoise


def shift(sig, interval=10):
    for col in range(sig.shape[1]):
        offset = np.random.choice(range(-interval, interval))
        sig[:, col] += offset
    return sig


def transform(sig, train=False):
    sig = resample(sig, config.target_point_num)
    np.random.seed()
    if np.random.rand() > 0.5: sig = scaling(sig)
    if np.random.rand() > 0.5: sig = shift(sig)
    sig = sig.transpose()
    sig = torch.tensor(sig.copy(), dtype=torch.float)
    return sig


class ECGDataset(Dataset):
    def __init__(self, data_path, train=True):
        super(ECGDataset, self).__init__()
        dd = torch.load(config.train_data)
        self.train = train
        self.data = dd['train'] if train else dd['val']
        self.idx2name = dd['idx2name']
        self.file2idx = dd['file2idx']
        self.wc = 1. / np.log(dd['wc'])

    def __getitem__(self, index):
        fid = self.data[index]
        file_path = os.path.join(config.train_dir, fid)
        df = pd.read_csv(file_path, sep=' ').values
        x = transform(df, self.train).reshape((1, 8, config.target_point_num))
        target = np.zeros(config.num_classes)
        target[self.file2idx[fid]] = 1
        target = torch.tensor(target, dtype=torch.float32)
        return x, target

    def __len__(self):
        return len(self.data)


if __name__ == '__main__':
    d = ECGDataset(config.train_data)
    print(d[0])
