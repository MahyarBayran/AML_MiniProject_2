import logging
import math
import os
import pickle
from collections import defaultdict
from copy import deepcopy

import numpy as np
import pandas as pd

from data import feature_extraction, pre_processing

logger = logging.getLogger(__name__)


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
    test_data = load_data('./dataset/test/')

    dataset = train_data + test_data

    return dataset


def top_information_gain(ds, fl):
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
    with open('./files/i_gain_{fl}.csv'.format(fl=fl), 'w') as f:
        for t, i in i_g_srt:
            f.write('{t},{i}\n'.format(t=t, i=i))

    fs = list(map(lambda x: x[0], i_g_srt))
    return fs


def extract_features(ds, tf_top_n, bf_top_n):
    ds_ef = deepcopy(ds)
    ds_ef = feature_extraction.top_n_frequency(ds_ef, 'tf', tf_top_n)
    ds_ef = feature_extraction.top_n_occurrence(ds_ef, 'tf', tf_top_n)
    ds_ef = feature_extraction.top_n_frequency(ds_ef, 'bf', bf_top_n)
    ds_ef = feature_extraction.top_n_occurrence(ds_ef, 'bf', bf_top_n)
    return ds_ef


def prepare_dataset(ds):
    ds_pd = pd.DataFrame.from_dict(ds, dtype=np.float64)
    y = ds_pd.get('label', None)
    x = ds_pd.drop(['label', 'text', 'text_pp', 'tf', 'bf'], axis=1)
    return x, y


def get_pre_processed_dataset():
    pre_processed_pickle_path = './files/pre_processed_dataset.pkl'
    if os.path.exists(pre_processed_pickle_path):
        with open(pre_processed_pickle_path, 'rb') as f:
            tr, ts = pickle.load(f)
    else:
        ds = load_dataset()
        ds_pp = pre_processing.pre_process(ds)
        tr, ts = split_dataset(ds_pp)
        with open(pre_processed_pickle_path, 'wb') as f:
            pickle.dump((tr, ts), f, pickle.HIGHEST_PROTOCOL)
    return tr, ts


def get_tf_top(ds):
    tf_top_pickle_path = './files/tf_top.pkl'
    if os.path.exists(tf_top_pickle_path):
        with open(tf_top_pickle_path, 'rb') as f:
            tf_top = pickle.load(f)
    else:
        tf_top = top_information_gain(ds, 'tf')
        with open(tf_top_pickle_path, 'wb') as f:
            pickle.dump(tf_top, f, pickle.HIGHEST_PROTOCOL)
    return tf_top


def get_bf_top(ds):
    bf_top_pickle_path = './files/bf_top.pkl'
    if os.path.exists(bf_top_pickle_path):
        with open(bf_top_pickle_path, 'rb') as f:
            bf_top = pickle.load(f)
    else:
        bf_top = top_information_gain(ds, 'bf')
        with open(bf_top_pickle_path, 'wb') as f:
            pickle.dump(bf_top, f, pickle.HIGHEST_PROTOCOL)
    return bf_top


def k_fold_cross_validation(k, cl, x_tr, y_tr):
    from sklearn.model_selection import cross_validate, KFold

    cv = KFold(n_splits=k, shuffle=True)
    scoring = ['f1']
    cv_scores = cross_validate(cl, x_tr.values, y_tr.values, cv=cv, scoring=scoring, return_train_score=True, n_jobs=-1)
    logger.info('KFold Cross Validation Scores: {cv_scores}'.format(cv_scores=cv_scores, indent=4))


def predict_test(cl, x_tr, y_tr, x_ts, fn):
    cl.fit(x_tr.values, y_tr.values)
    with open('./results/{fn}.csv'.format(fn=fn), 'w') as f:
        f.write('Id,Category\n')
        for i, y_i in enumerate(cl.predict(x_ts.values)):
            f.write('{i},{y_i}\n'.format(i=i, y_i=int(y_i)))
