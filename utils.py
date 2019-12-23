import torch
import numpy as np
import time, os
from sklearn.metrics import f1_score
from torch import nn


def mkdirs(path):
    if not os.path.exists(path):
        os.makedirs(path)


def calc_f1(y_true, y_pre, threshold=0.5):
    y_true = y_true.view(-1).cpu().detach().numpy().astype(np.int)
    y_pre = y_pre.view(-1).cpu().detach().numpy() > threshold
    return f1_score(y_true, y_pre)


def print_time_cost(since):
    time_elapsed = time.time() - since
    return '{:.0f}m{:.0f}s\n'.format(time_elapsed // 60, time_elapsed % 60)


def adjust_learning_rate(optimizer, lr):
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return lr


class WeightedMultilabel(nn.Module):
    def __init__(self, weights):
        super(WeightedMultilabel, self).__init__()
        self.cerition = nn.BCEWithLogitsLoss(reduction='none')
        self.weights = weights
        for i in [0, 2, 6, 7, 8, 14, 15, 16, 19, 21, 23, 25, 32, 33]:
            self.weights[i] = 0
        self.alpha = 2.5
        self.gamma = 2

    def forward(self, outputs, targets):
        loss = self.cerition(outputs, targets)
        pt = torch.exp(-loss)
        loss = self.alpha * (1 - pt) ** self.gamma * loss
        return (loss * self.weights).sum()
