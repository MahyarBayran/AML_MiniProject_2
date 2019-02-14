import logging
from datetime import datetime

import numpy as np
from sklearn.base import clone
from sklearn.feature_selection import mutual_info_classif
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import BernoulliNB, GaussianNB

from data import helpers

logger = logging.getLogger(__name__)


def setup():
    np.random.seed(2019)

    log_path = datetime.now().strftime('./logs/%Y-%m-%d-%H-%M-%S.log')
    logging.basicConfig(filename=log_path, level=logging.INFO)
    logging.getLogger().addHandler(logging.StreamHandler())


def gaussian_naive_bayes(x_tr, y_tr, x_ts):
    cl = GaussianNB()
    helpers.k_fold_cross_validation(4, clone(cl), x_tr, y_tr)
    helpers.predict_test(clone(cl), x_tr, y_tr, x_ts, 'gaussian_naive_bayes')


def bernoulli_naive_bayes(x_tr, y_tr, x_ts):
    cl = BernoulliNB()
    helpers.k_fold_cross_validation(4, clone(cl), x_tr, y_tr)
    helpers.predict_test(clone(cl), x_tr, y_tr, x_ts, 'bernoulli_naive_bayes')


def logistic_regression(x_tr, y_tr, x_ts):
    cl = LogisticRegression(solver='lbfgs', n_jobs=-1)
    helpers.k_fold_cross_validation(4, clone(cl), x_tr, y_tr)
    helpers.predict_test(clone(cl), x_tr, y_tr, x_ts, 'logistic_regression')


def main():
    logger.info('[{t}] Start'.format(t=datetime.now()))
    x_tr, y_tr, x_ts = helpers.load_dataset()
    logger.info('[{t}] Dataset Loaded'.format(t=datetime.now()))
    x_tr, x_ts, fn = helpers.extract_features(x_tr, x_ts)
    logger.info('[{t}] Features Extracted'.format(t=datetime.now()))
    mi = mutual_info_classif(x_tr, y_tr)
    logger.info('[{t}] Mutual Info Calculated'.format(t=datetime.now()))

    ig = np.array(sorted(zip(mi, fn), key=lambda x: -x[0]))
    with open('./files/info_gain.csv', 'w') as f:
        for mi_i, fn_i in ig:
            f.write('{mi_i},{fn_i}\n'.format(mi_i=mi_i, fn_i=fn_i))

    logger.info('[{t}] Gaussian Naive Bayes'.format(t=datetime.now()))
    gaussian_naive_bayes(x_tr.toarray(), y_tr, x_ts.toarray())
    logger.info('[{t}] Bernoulli Naive Bayes'.format(t=datetime.now()))
    bernoulli_naive_bayes(x_tr.toarray(), y_tr, x_ts.toarray())
    logger.info('[{t}] Logistic Regression'.format(t=datetime.now()))
    logistic_regression(x_tr.toarray(), y_tr, x_ts.toarray())
    logger.info('[{t}] Finish'.format(t=datetime.now()))


if __name__ == '__main__':
    setup()
    main()
