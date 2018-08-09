from scipy import stats

class UniformDistribution:
    def __init__(self, min, max):
        # We introduce 1% of "error" to avoid problem with limit cases
        self.min = min - min * 0.01
        self.max = max + max * 0.01
        self.scale = self.max - self.min

    def draw(self):
        return stats.uniform.rvs(loc=self.min, scale=self.scale)

    def cdf(self, x):
        return stats.uniform.cdf(x, loc=self.min, scale=self.scale)

    def inverse_cdf(self, p):
        return stats.uniform.ppf(p, loc=self.min, scale=self.scale)
