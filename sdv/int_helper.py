from sdv.float_helper import FloatHelper


class IntHelper:

    def __init__(self, sample):
        self.float_helper = FloatHelper(sample)

    def gaussian_copula(self, x):
        return self.float_helper.gaussian_copula(x)

    def inverse_gaussian_copula(self, x):
        return self.float_helper.inverse_gaussian_copula(x)

    def preprocess(self, x):
        return int(x)

    def postprocess(self, x):
        return round(x)