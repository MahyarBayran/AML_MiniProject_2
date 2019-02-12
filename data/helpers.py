import math
import os
from collections import defaultdict


def split_dataset(ds):
    return ds[:25000], ds[25000:]


def load_data(base_path, label=None):
    data = []
    for f_name in os.listdir(base_path):
        with open('{base_path}{f_name}'.format(base_path=base_path, f_name=f_name), 'r') as f:
            data.append({'text': f.read(), 'label': label})
    return data


def load_dataset():
    pos_data = load_data('./dataset/train/pos/', 1)
    neg_data = load_data('./dataset/train/neg/', 0)

    train_data = neg_data + pos_data
    test_data = load_data('test/')

    dataset = train_data + test_data

    return dataset


def top_n_information_gain(ds, fl, n):
    cnt_y = defaultdict(int)
    cnt_xj = defaultdict(int)
    cnt_xj_y = defaultdict(int)
    for d in ds:
        for t, c in d[fl].items():
            cnt_y[d['label']] += 1
            cnt_xj[(t, c)] += 1
            cnt_xj_y[(t, c, d['label'])] += 1

    i_g = defaultdict(int)
    for (t, c), v in cnt_xj.items():
        p_xj = v / sum(cnt_y.values())
        for y in (0, 1):
            p_y = cnt_y[y] / sum(cnt_y.values())
            p_xj_y = cnt_xj_y[(t, c, y)] / sum(cnt_y.values())
            if p_xj_y > 0:
                i_g[t] += p_xj_y * math.log2(p_xj_y / (p_xj * p_y))

    i_g_srt = sorted(i_g.items(), key=lambda x: -x[1])
    with open('i_gain_{fl}.csv'.format(fl=fl), 'w') as f:
        for t, i in i_g_srt:
            f.write('{t},{i}\n'.format(t=t, i=i))

    fs = list(map(lambda x: x[0], i_g_srt))[:n]
    return fs
