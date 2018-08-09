import logging
import warnings

from scipy import stats

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

    def gaussian_copula(self, x):
        return stats.norm.ppf(self.distribution.cdf(x))

    def inverse_gaussian_copula(self, x):
        return self.distribution.inverse_cdf(stats.norm.cdf(x))

    def preprocess(self, x):
        return float(x)

    def postprocess(self, x):
        return x

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
