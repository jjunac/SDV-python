import numpy as np

import sdv.data_type as dt
import scipy.stats
from collections import Counter

from sdv.categorical_helper import CategoricalHelper
from sdv.float_helper import FloatHelper
from sdv.int_helper import IntHelper


def syn(metadata, data, size=1):
    # Pre-processing
    helpers = __compute_helpers(metadata, data)
    processed_data = __preprocess(helpers, data)

    # Computing covariance
    # y = [[helpers[column].gaussian_copula(value) for column, value in enumerate(row)] for row in processed_data]
    # cov_matrix = np.cov(y, rowvar=False)
    cov_matrix = np.corrcoef(processed_data, rowvar=False)

    # Sampling
    randoms = [np.random.normal(size=len(metadata)) for i in range(size)]
    l = np.linalg.cholesky(cov_matrix)
    samples = [np.dot(l, v) for v in randoms]

    # Convert back to original space and postprocess
    results = [[helpers[column].inverse_gaussian_copula(value) for column, value in enumerate(row)] for row in samples]
    postprocessed_results = __postprocess(helpers, results)

    return postprocessed_results


def __compute_helpers(metadata, data):
    helpers = []
    for column, type in enumerate(metadata):
        sample = [row[column] for row in data]

        if type == dt.CATEGORICAL:
            helper = CategoricalHelper(sample)
            helpers.append(helper)
        elif type == dt.FLOAT:
            float_sample = list(map(float, sample))
            helpers.append(FloatHelper(float_sample))
        elif type == dt.INT:
            int_sample = list(map(int, sample))
            helpers.append(IntHelper(int_sample))
        else:
            raise RuntimeError("Unknown type")
    return helpers


def __preprocess(helpers, data):
    return [[helpers[column].preprocess(value) for column, value in enumerate(row)] for row in data]


def __postprocess(helpers, generated_data):
    return [[helpers[column].postprocess(value) for column, value in enumerate(row)] for row in generated_data]
