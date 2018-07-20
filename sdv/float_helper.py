from scipy import stats

from sdv.normal_distribution import NormalDistribution
from sdv.uniform_distribution import UniformDistribution


class FloatHelper:

    def __init__(self, sample):
        loc, scale = stats.norm.fit(sample)
        # create a normal distribution with loc and scale
        n = stats.norm(loc=loc, scale=scale)

        p_value_norm = stats.kstest(sample, n.cdf)
        p_value_uniform = stats.kstest(sample, 'uniform')
        if p_value_uniform.pvalue > p_value_norm.pvalue:
            self.distribution = UniformDistribution(min(sample), max(sample))
        else:
            self.distribution = NormalDistribution(loc, scale, min(sample), max(sample))

    def gaussian_copula(self, x):
        return stats.norm.ppf(self.distribution.cdf(x))

    def inverse_gaussian_copula(self, x):
        return self.distribution.inverse_cdf(stats.norm.cdf(x))

    def preprocess(self, x):
        return float(x)

    def postprocess(self, x):
        return x