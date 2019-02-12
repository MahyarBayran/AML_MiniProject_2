import logging
import os
import pickle
from copy import deepcopy
from datetime import datetime

import numpy as np
import pandas as pd

from data import helpers, pre_processing, feature_extraction

logger = logging.getLogger(__name__)


def setup_logger():
    log_path = datetime.now().strftime('./logs/%Y-%m-%d-%H-%M-%S.log')
    logging.basicConfig(filename=log_path, level=logging.INFO)
    logging.getLogger().addHandler(logging.StreamHandler())


def setup_numpy():
    np.random.seed(2019)


def extract_features(ds, tf_top_n, bf_top_n):
    ds_ef = deepcopy(ds)
    ds_ef = feature_extraction.top_n_term_frequency(ds_ef, tf_top_n)
    ds_ef = feature_extraction.top_n_term_occurrence(ds_ef, tf_top_n)
    ds_ef = feature_extraction.top_n_term_frequency(ds_ef, bf_top_n)
    ds_ef = feature_extraction.top_n_term_occurrence(ds_ef, bf_top_n)
    return ds_ef


def prepare_dataset(ds):
    ds_pd = pd.DataFrame.from_dict(ds, dtype=np.float64)
    y = ds_pd.get('label', None)
    x = ds_pd.drop(['text', 'text_pp', 'tf', 'bf'], axis=1)
    return x, y


def main():
    setup_logger()
    setup_numpy()

    pre_processed_pickle_path = './files/pre_processed_dataset.pkl'
    if os.path.exists(pre_processed_pickle_path):
        with open(pre_processed_pickle_path, 'rb') as f:
            tr, ts = pickle.load(f)
    else:
        ds = helpers.load_dataset()
        ds_pp = pre_processing.pre_process(ds)
        tr, ts = helpers.split_dataset(ds_pp)
        with open(pre_processed_pickle_path, 'wb') as f:
            pickle.dump((tr, ts), f, pickle.HIGHEST_PROTOCOL)

    tf_top_n_pickle_path = './files/tf_top_n.pkl'
    if os.path.exists(tf_top_n_pickle_path):
        with open(tf_top_n_pickle_path, 'rb') as f:
            tf_top_n = pickle.load(f)
    else:
        tf_top_n = helpers.top_n_information_gain(tr, 'tf', 100)
        with open(tf_top_n_pickle_path, 'wb') as f:
            pickle.dump(tf_top_n, f, pickle.HIGHEST_PROTOCOL)

    bf_top_n_pickle_path = './files/bf_top_n.pkl'
    if os.path.exists(bf_top_n_pickle_path):
        with open(bf_top_n_pickle_path, 'rb') as f:
            bf_top_n = pickle.load(f)
    else:
        bf_top_n = helpers.top_n_information_gain(tr, 'bf', 100)
        with open(bf_top_n_pickle_path, 'wb') as f:
            pickle.dump(bf_top_n, f, pickle.HIGHEST_PROTOCOL)

    tr, ts = list(map(lambda x: extract_features(x, tf_top_n, bf_top_n), (tr, ts)))
    (x_tr, y_tr), (x_ts, y_ts) = list(map(lambda x: prepare_dataset(x), (tr, ts)))


if __name__ == '__main__':
    main()
