from scipy import stats


class ExponentialDistribution:
    def __init__(self, mean, variance):
        self.mean = mean
        self.variance = variance

    def draw(self):
        return stats.expon.rvs(loc=self.mean, scale=self.variance)

    def cdf(self, arr):
        return stats.expon.cdf(arr, loc=self.mean, scale=self.variance)

    def inverse_cdf(self, arr):
        return stats.expon.ppf(arr, loc=self.mean, scale=self.variance)
