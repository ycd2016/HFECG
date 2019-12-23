import pandas as pd
from sklearn.preprocessing import MultiLabelBinarizer
import numpy as np
from tqdm import tqdm
import biosppy
import os
import warnings
import pyentrp.entropy as ent
from scipy.stats import skew, kurtosis
from scipy.signal import resample
from sklearn.preprocessing import minmax_scale, scale
from config import config
warnings.filterwarnings('ignore')


def get_gender(x):
    if x == 'FEMALE':
        return 0
    elif x == 'MALE':
        return 1
    else:
        return np.nan


def get_label(tmp_label, file_name):
    tmp_list_1 = []
    tmp_list_2 = []
    mlb = MultiLabelBinarizer(classes=tmp_label)
    with open(file_name, encoding='utf8') as f:
        for line in f.readlines():
            label_tmp = line.strip().split('\t')
            tmp_list_1.append(label_tmp[:3])
            tmp_list_2.append(label_tmp[3:])
    label_df = mlb.fit_transform(tmp_list_2)
    tmp_df = pd.DataFrame(tmp_list_1, columns=['filename', 'age', 'gender'])
    tmp_df.replace('', np.nan, inplace=True)
    tmp_df['age'] = tmp_df['age'].astype(float)
    tmp_df['gender'] = tmp_df['gender'].apply(get_gender)
    label_df = pd.DataFrame(label_df, columns=tmp_label)
    return pd.concat([tmp_df, label_df], axis=1)


def gen_data(file_list, fs, resample_num):
    tmp_array = []
    key = ['I', 'II', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6']
    for t in tqdm(range(len(file_list))):
        tmp_list = []
        s = file_list[t]
        tmp = pd.read_csv(s, sep=' ', engine='python')
        for i, z in enumerate(key):
            try:
                tmp_lead = biosppy.ecg.ecg(tmp[z],
                                           show=False,
                                           sampling_rate=fs)
            except:
                pass
            rpeaks = tmp_lead['rpeaks']
            if rpeaks.shape[0] != 0:
                rr_intervals = np.diff(rpeaks)
                min_dis = rr_intervals.min()
                drr = np.diff(rr_intervals)
                r_density = (rr_intervals.shape[0] + 1) / tmp[z].shape[0] * fs
                pnn50 = drr[drr >= fs * 0.05].shape[0] / rr_intervals.shape[0]
                rmssd = np.sqrt(np.mean(drr * drr))
                samp_entrp = ent.sample_entropy(rr_intervals, 2,
                                                0.2 * np.std(rr_intervals))
                samp_entrp[np.isnan(samp_entrp)] = -2
                samp_entrp[np.isinf(samp_entrp)] = -1
                tmp_list.extend(
                    [rr_intervals.min(), rr_intervals.max(),
                     rr_intervals.mean(), rr_intervals.std(),
                     skew(rr_intervals), kurtosis(rr_intervals), r_density,
                     pnn50, rmssd, samp_entrp[0], samp_entrp[1]])
            else:
                tmp_list.extend([np.nan] * 11)
            heart_rate = tmp_lead['heart_rate']
            if heart_rate.shape[0] != 0:
                tmp_list.extend([heart_rate.min(), heart_rate.max(),
                                 heart_rate.mean(), heart_rate.std(),
                                 skew(heart_rate), kurtosis(heart_rate)])
            else:
                tmp_list.extend([np.nan] * 6)
            templates = tmp_lead['templates']
            templates_min = templates.min(axis=0)
            templates_max = templates.max(axis=0)
            templates_diff = templates_max - templates_min
            templates_mean = templates.mean(axis=0)
            templates_std = templates.std(axis=0)
            for j in [templates_diff, templates_mean, templates_std]:
                tmp_rmp = resample(j, num=resample_num)
                tmp_list.extend(list(tmp_rmp))
        tmp_array.append(tmp_list)
    tmp_df = pd.DataFrame(tmp_array)
    return tmp_df


test = pd.read_csv(config.test_label,
                   sep='\t',
                   header=None,
                   names=['filename', 'age', 'gender'])
path_all = [os.path.join(config.test_dir, tmp_file)
            for tmp_file in test['filename']]
test_handle_df = gen_data(path_all, 500, 50)
test_handle_df.to_csv('feature.csv', index=False, encoding='utf-8')
