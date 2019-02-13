import logging
from datetime import datetime

import numpy as np

from data import helpers


def setup():
    np.random.seed(2019)

    log_path = datetime.now().strftime('./logs/%Y-%m-%d-%H-%M-%S.log')
    logging.basicConfig(filename=log_path, level=logging.INFO)
    logging.getLogger().addHandler(logging.StreamHandler())


def gaussian_naive_bayes(x_tr, y_tr, x_ts):
    from sklearn.naive_bayes import GaussianNB
    from sklearn.base import clone

    cl = GaussianNB()
    helpers.k_fold_cross_validation(4, clone(cl), x_tr, y_tr)
    helpers.predict_test(clone(cl), x_tr, y_tr, x_ts, 'gaussian_naive_bayes')


def bernoulli_naive_bayes(x_tr, y_tr, x_ts):
    from sklearn.naive_bayes import BernoulliNB
    from sklearn.base import clone

    cl = BernoulliNB()
    helpers.k_fold_cross_validation(4, clone(cl), x_tr, y_tr)
    helpers.predict_test(clone(cl), x_tr, y_tr, x_ts, 'bernoulli_naive_bayes')


def main():
    tr, ts = helpers.get_pre_processed_dataset()
    tf_top = helpers.get_tf_top(tr)
    bf_top = helpers.get_bf_top(tr)

    tr, ts = list(map(lambda x: helpers.extract_features(x, tf_top[:40], bf_top[:15]), (tr, ts)))
    (x_tr, y_tr), (x_ts, _) = list(map(lambda x: helpers.prepare_dataset(x), (tr, ts)))

    gaussian_naive_bayes(x_tr, y_tr, x_ts)
    bernoulli_naive_bayes(x_tr, y_tr, x_ts)


if __name__ == '__main__':
    setup()
    main()
