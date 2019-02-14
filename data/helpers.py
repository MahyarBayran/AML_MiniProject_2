import logging
import os

import numpy as np
from sklearn.model_selection import cross_validate, KFold

from data import feature_extraction

logger = logging.getLogger(__name__)


def load_data(dp):
    x = []
    for fn in os.listdir(dp):
        with open('{dp}{fn}'.format(dp=dp, fn=fn), 'r') as f:
            x.append(f.read())
    return x


def load_dataset():
    x_tr_pos = np.array(load_data('./dataset/train/pos/'), dtype=np.str)
    y_tr_pos = np.ones_like(x_tr_pos, dtype=np.float64)

    x_tr_neg = np.array(load_data('./dataset/train/neg/'), dtype=np.str)
    y_tr_neg = np.zeros_like(x_tr_neg, dtype=np.float64)

    x_tr = np.concatenate((x_tr_pos, x_tr_neg), axis=0)
    y_tr = np.concatenate((y_tr_pos, y_tr_neg), axis=0)

    x_ts = np.array(load_data('./dataset/test/'), dtype=np.str)

    return x_tr, y_tr, x_ts


def extract_features(x_tr, x_ts):
    x_tr, tiv = feature_extraction.tf_idf_vectorizer(x_tr, ngram_range=(1, 3), use_idf=False, binary=True)
    x_ts = tiv.transform(x_ts)
    fn = np.array(tiv.get_feature_names(), dtype=np.str)
    return x_tr, x_ts, fn


def k_fold_cross_validation(k, cl, x_tr, y_tr):
    cv = KFold(n_splits=k, shuffle=True)
    cv_s = cross_validate(cl, x_tr, y_tr, cv=cv, scoring='f1', return_train_score=True, n_jobs=-1)
    logger.info('KFold Cross Validation Scores: {cv_s}'.format(cv_s=cv_s, indent=4))


def predict_test(cl, x_tr, y_tr, x_ts, fn):
    cl.fit(x_tr, y_tr)
    with open('./results/{fn}.csv'.format(fn=fn), 'w') as f:
        f.write('Id,Category\n')
        for i, y_i in enumerate(cl.predict(x_ts)):
            f.write('{i},{y_i}\n'.format(i=i, y_i=int(y_i)))
