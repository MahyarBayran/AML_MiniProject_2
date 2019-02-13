import math
from collections import defaultdict

import numpy as np


class BernoulliNB(object):
    def __init__(self):
        self._t = None
        self._t_y = defaultdict(int)
        self._t_x_y = defaultdict(lambda: 1)  # Note: Laplace Smoothing

    def fit(self, x, y):
        assert self._t is None
        assert len(self._t_y) == 0
        assert len(self._t_x_y) == 0

        for x_i, y_i in zip(x, y):
            self._t_y[y_i] += 1
            for j, x_i_j in enumerate(x_i):
                self._t_x_y[(j, x_i_j, y_i)] += 1

        self._t = sum(self._t_y.values())

        for (j, x_i_j, y_i) in self._t_x_y.keys():
            self._t_x_y[(j, x_i_j, y_i)] /= self._t_y[y_i] + 2

        for y_i in self._t_y.keys():
            self._t_y[y_i] /= self._t

    def predict(self, x):
        assert self._t is not None
        assert len(self._t_y) > 0
        assert len(self._t_x_y) > 0

        pr = []
        for x_i in x:
            lo = math.log2(self._t_y[1] / (1 - self._t_y[1]))
            for j, x_i_j in enumerate(x_i):
                lo += x_i_j * math.log2(self._t_x_y[(j, x_i_j, 1)] / self._t_x_y[(j, x_i_j, 0)])
                lo += (1 - x_i_j) * math.log2((1 - self._t_x_y[(j, x_i_j, 1)]) / (1 - self._t_x_y[(j, x_i_j, 0)]))
            pr.append(int(lo >= 0))
        return np.array(pr, dtype=np.float64)
