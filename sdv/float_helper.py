from scipy import stats

from sdv.beta_distribution import BetaDistribution
from sdv.exponential_distribution import ExponentialDistribution
from sdv.normal_distribution import NormalDistribution
from sdv.uniform_distribution import UniformDistribution


class FloatHelper:

    def __init__(self, sample):
        distributions = self.__compute_distributions(sample)
        best_distrib = max(distributions, key=lambda x: distributions[x]["pvalue"])
        if best_distrib == "uniform":
            self.distribution = UniformDistribution(*distributions[best_distrib]['args'])
        elif best_distrib == "normal":
            self.distribution = NormalDistribution(*distributions[best_distrib]['args'])
        elif best_distrib == "expon":
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

        # Uniform
        res['uniform'] = {'pvalue': stats.kstest(sample, 'uniform').pvalue, 'args': [mini, maxi]}
        # Normal
        loc, scale = stats.norm.fit(sample)
        res['normal'] = {'pvalue': stats.kstest(sample, stats.norm(loc=loc, scale=scale).cdf).pvalue, 'args': [loc, scale, mini, maxi]}
        # Exponential
        loc, scale = stats.expon.fit(sample)
        res['expon'] = {'pvalue': stats.kstest(sample, stats.expon(loc=loc, scale=scale).cdf).pvalue, 'args': [loc, scale]}
        # Beta
        a, b, loc, scale = stats.beta.fit(sample)
        res['beta'] = {'pvalue': stats.kstest(sample, stats.beta(a=a, b=b, loc=loc, scale=scale).cdf).pvalue, 'args': [loc, scale, a, b]}

        return res
