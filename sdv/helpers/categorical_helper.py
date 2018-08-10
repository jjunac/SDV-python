import logging
from collections import Counter
from scipy import stats
import random
import numpy as np


PRECISION_BACKTRACKING_MAP = 6

DELTA_BACKTRACKING_MAP = 5 * 10**-(PRECISION_BACKTRACKING_MAP+1)
STEP_BACKTRACKING_MAP = 10**-PRECISION_BACKTRACKING_MAP


class CategoricalHelper:
    def __init__(self, sample, column_name):
        self.bounds = {}
        self.draws = {}
        self.class_finder = {}

        # We analyze frequency of every class and allocate them a range in the [0, 1) segment,
        # as described in the Synthetic Data Vault Paper
        counter = Counter(sample)
        cumulative_probability = 0
        for clazz, nb in counter.most_common():
            p = nb / len(sample)
            max_bound = cumulative_probability + p
            self.bounds[clazz] = (cumulative_probability, max_bound)
            # For performance issues, we are anticipating the pre-processing draws
            self.draws[clazz] = self.draw_for_class(clazz, size=nb).tolist() if nb > 1 else [self.draw_for_class(clazz)]
            # For performance issues, we generating a map to improve class backtracking
            q = np.around(cumulative_probability + DELTA_BACKTRACKING_MAP, PRECISION_BACKTRACKING_MAP)
            while q < max_bound:
                self.class_finder[q] = clazz
                q = np.around(q + STEP_BACKTRACKING_MAP, PRECISION_BACKTRACKING_MAP)

            cumulative_probability += p

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
        def find_class(x):
            c1 = self.class_finder[np.round(x - DELTA_BACKTRACKING_MAP, PRECISION_BACKTRACKING_MAP)]
            c2 = self.class_finder[np.round(x + DELTA_BACKTRACKING_MAP, PRECISION_BACKTRACKING_MAP)]
            if c1 == c2 or x < self.bounds[c1][1]:
                return c1
            return c2
        return np.array(list(map(find_class, arr)))
