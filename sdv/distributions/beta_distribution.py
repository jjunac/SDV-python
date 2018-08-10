from scipy import stats


class BetaDistribution:
    def __init__(self, mean, variance, a, b):
        self.mean = mean
        self.variance = variance
        self.a = a
        self.b = b

    def draw(self):
        return stats.beta.rvs(self.a, self.b, loc=self.mean, scale=self.variance)

    def cdf(self, arr):
        return stats.beta.cdf(arr, self.a, self.b, loc=self.mean, scale=self.variance)

    def inverse_cdf(self, arr):
        return stats.beta.ppf(arr, self.a, self.b, loc=self.mean, scale=self.variance)
