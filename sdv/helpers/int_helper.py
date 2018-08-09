from sdv.helpers.float_helper import FloatHelper


class IntHelper:

    def __init__(self, sample, column_name):
        self.float_helper = FloatHelper(sample, column_name)

    def gaussian_copula(self, x):
        return self.float_helper.gaussian_copula(x)

    def inverse_gaussian_copula(self, x):
        return self.float_helper.inverse_gaussian_copula(x)

    def preprocess(self, x):
        return int(float(x if x else 0))

    def postprocess(self, x):
        return int(round(x))
