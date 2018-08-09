from scipy import stats

class UniformDistribution:
    def __init__(self, min, max):
        # We introduce 1% of "error" to avoid problem with limit cases
        self.min = min - min * 0.01
        self.max = max + max * 0.01

    def draw(self):
        return self.__denormalise(stats.uniform.rvs())

    def cdf(self, x):
        return stats.uniform.cdf(self.__normalise(x))

    def inverse_cdf(self, p):
        return self.__denormalise(stats.uniform.ppf(p, self.min, self.max))

    def __denormalise(self, x):
        return self.min + x * (self.max - self.min)

    def __normalise(self, x):
        return (-self.min + x) / (self.max - self.min)