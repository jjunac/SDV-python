from scipy import stats


class NormalDistribution:
    def __init__(self, mean, variance, min, max):
        self.mean = mean
        self.variance = variance
        # We introduce 1% of "error" to avoid problem with limit cases
        self.min = min - min * 0.01
        self.max = max + max * 0.01
        # Because python, cf. https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.truncnorm.html
        self.normalised_min = (self.min - mean) / variance
        self.normalised_max = (self.max - mean) / variance

    def draw(self):
        return stats.truncnorm.rvs(self.normalised_min, self.normalised_max, loc=self.mean, scale=self.variance)

    def cdf(self, arr):
        return stats.truncnorm.cdf(arr, self.normalised_min, self.normalised_max, loc=self.mean, scale=self.variance)

    def inverse_cdf(self, arr):
        return stats.truncnorm.ppf(arr, self.normalised_min, self.normalised_max, loc=self.mean, scale=self.variance)
