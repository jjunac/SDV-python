import logging
import warnings

from scipy import stats
import numpy as np

from sdv.distributions import *


class FloatHelper:

    def __init__(self, sample, column_name):
        # Find the best fitting distribution using the Kolmogorov-Smirnov test
        distributions = self.__compute_distributions(sample)

        best_distrib = max(distributions, key=lambda x: distributions[x]["pvalue"])
        logging.debug("Best distribution for %s is the %s distribution with parameters %s" % (column_name, best_distrib, str(tuple(distributions[best_distrib]['args']))))
        if best_distrib == "uniform":
            self.distribution = UniformDistribution(*distributions[best_distrib]['args'])
        elif best_distrib == "normal":
            self.distribution = NormalDistribution(*distributions[best_distrib]['args'])
        elif best_distrib == "exponential":
            self.distribution = ExponentialDistribution(*distributions[best_distrib]['args'])
        elif best_distrib == "beta":
            self.distribution = BetaDistribution(*distributions[best_distrib]['args'])

    def gaussian_copula(self, arr):
        return stats.norm.ppf(self.distribution.cdf(arr))

    def inverse_gaussian_copula(self, arr):
        return self.distribution.inverse_cdf(stats.norm.cdf(arr))

    def preprocess(self, arr):
        return np.array([float(x) if x else 0.0 for x in arr])

    def postprocess(self, arr):
        return arr

    def __compute_distributions(self, sample):
        res = {}
        mini, maxi = min(sample), max(sample)

        # Ignore warnings for the Kolmogorov-Smirnov test, because the impossible distribution will throw a lot of them
        warnings.simplefilter('ignore')

        # Uniform
        res['uniform'] = {'pvalue': stats.kstest(sample, 'uniform').pvalue, 'args': [mini, maxi]}
        # Normal
        loc, scale = stats.norm.fit(sample)
        res['normal'] = {'pvalue': stats.kstest(sample, stats.norm(loc=loc, scale=scale).cdf).pvalue, 'args': [loc, scale, mini, maxi]}
        # Exponential
        loc, scale = stats.expon.fit(sample)
        res['exponential'] = {'pvalue': stats.kstest(sample, stats.expon(loc=loc, scale=scale).cdf).pvalue, 'args': [loc, scale]}
        # Beta
        a, b, loc, scale = stats.beta.fit(sample)
        res['beta'] = {'pvalue': stats.kstest(sample, stats.beta(a=a, b=b, loc=loc, scale=scale).cdf).pvalue, 'args': [loc, scale, a, b]}

        # Bring the warnings back
        warnings.simplefilter('default')

        return res
