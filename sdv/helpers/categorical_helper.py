import logging
from collections import Counter
from scipy import stats
import random
import numpy as np
import bisect

class CategoricalHelper:
    def __init__(self, sample, column_name):
        self.bounds = {}
        self.draws = {}
        self.lower_bounds = []
        self.decreasing_order = []

        # We analyze frequency of every class and allocate them a range in the [0, 1) segment,
        # as described in the Synthetic Data Vault Paper
        counter = Counter(sample)
        cumulative_probability = 0
        for clazz, nb in counter.most_common():
            p = nb / len(sample)
            self.lower_bounds.append(cumulative_probability)
            self.decreasing_order.append(clazz)
            self.bounds[clazz] = (cumulative_probability, cumulative_probability + p)
            cumulative_probability += p
            # For performance issues, we are anticipating the pre-processing draws
            self.draws[clazz] = self.draw_for_class(clazz, size=nb).tolist() if nb > 1 else [self.draw_for_class(clazz)]
        logging.debug("Range for %s are %s" % (column_name, str(self.bounds)))

    def draw_for_class(self, clazz, size=1):
        # Draw a float in the range allocated to the class
        bounds = self.bounds[clazz]
        mean = bounds[0] + (bounds[1] - bounds[0]) / 2
        variance = (bounds[1] - bounds[0]) / 6
        # Because python, see https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.truncnorm.html
        normalised_min = (bounds[0] - mean) / variance
        normalised_max = (bounds[1] - mean) / variance
        res = stats.truncnorm.rvs(normalised_min, normalised_max, loc=mean, scale=variance, size=size)
        return res if size > 1 else res[0]

    def draw_a_class(self):
        return self.postprocess(random.random())

    def gaussian_copula(self, arr):
        return stats.norm.ppf(stats.uniform.cdf(arr))

    def inverse_gaussian_copula(self, arr):
        return stats.norm.cdf(arr)

    def preprocess(self, arr):
        return np.array([self.draws[x].pop() for x in arr])

    def postprocess(self, arr):
        return np.array([self.decreasing_order[bisect.bisect_right(self.lower_bounds, x)-1] for x in arr])
