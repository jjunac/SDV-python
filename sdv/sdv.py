import logging
from collections import Counter
from datetime import datetime
import time

import numpy as np

import sdv.data_type as dt
from sdv.helpers import *

logging.basicConfig(format='%(asctime)s\t[%(levelname)-5s] %(message)s', datefmt='%Y-%m-%d %H:%M:%S', level=logging.DEBUG)


def syn(metadata, data, size=1, header=None):
    stime = time.time()
    ndata = np.array(data)
    # Pre-processing
    logging.info("Analyzing distributions")
    helpers = __compute_helpers(metadata, data, header)
    logging.info("Pre-processing")
    processed_data = __preprocess(helpers, ndata)

    # Computing Pearson correlation coefficient matrix
    logging.info("Computing Pearson correlation coefficient matrix")
    cov_matrix = np.corrcoef(processed_data, rowvar=False)
    logging.info("Applying Cholesky factorization")
    l = np.linalg.cholesky(cov_matrix)

    # Sampling
    logging.info("Generating %d rows" % size)
    randoms = np.random.normal(size=(size, len(metadata)))
    logging.info("Applying correlation factors")
    samples = np.apply_along_axis(lambda v: np.dot(l, v), 1, randoms)

    # Convert back to original space and post-process
    logging.info("Converting data back to original space")
    results = __apply_helpers(helpers, samples, lambda h, col: h.inverse_gaussian_copula(col))
    logging.info("Post-processing")
    postprocessed_results = __postprocess(helpers, results)

    ttime = int(time.time() - stime)
    logging.info("Synthesized %s rows in %dh%02dm%02ds" % (size, ttime//3600, (ttime//60) % 60, ttime % 60))
    return postprocessed_results


def syn_by_class(metadata, data, class_column, size=1, header=None):
    # Analyze the distribution of the class values
    class_name = header[class_column] if header else 'class'
    logging.info("Analyzing %s distribution" % (class_name))
    categorical_helper = CategoricalHelper([row[class_column] for row in data], class_name)

    # Synthesize the class values that we'll use to determine the other values
    logging.info("Drawing %s values" % class_name)
    class_draws = [categorical_helper.draw_a_class() for _ in range(size)]
    # Count the occurrence of a value
    counter = Counter(class_draws)
    # Remove the class from the metadata and header, so that we can synthesize the other value
    metadata_wo_cat = metadata[0:class_column] + metadata[class_column+1:len(metadata)]
    header_wo_cat = header[0:class_column] + header[class_column+1:len(header)] if header else None
    res = []

    # For every class, we take the corresponding rows and we analyze them in order to synthesize them
    for clazz, n in dict(counter).items():
        logging.info("Synthesizing %d rows of class %s" % (n, clazz))
        sample = [e[0:class_column] + e[class_column+1:len(e)] for e in data if e[class_column] == clazz]
        res_for_class = syn(metadata_wo_cat, sample, size=n, header=header_wo_cat)
        for row in res_for_class:
            row.insert(class_column, clazz)
        res.extend(res_for_class)
    return res


def __compute_helpers(metadata, data, header=None):
    helpers = []
    for column, type in enumerate(metadata):
        sample = [row[column] for row in data]
        column_name = header[column] if header else 'column_%d' % column

        # Find the helper for the specified data type
        # The helper will analyse the distribution and contain all the information needed to apply the gaussian copula etc
        if type == dt.CATEGORICAL:
            logging.debug("Use categorical helper for '%s'" % column_name)
            helper = CategoricalHelper(sample, column_name)
            helpers.append(helper)
        elif type == dt.FLOAT:
            logging.debug("Use float helper for '%s'" % column_name)
            float_sample = [float(e if e else 0) for e in sample]
            helpers.append(FloatHelper(float_sample, column_name))
        elif type == dt.INT:
            logging.debug("Use int helper for '%s'" % column_name)
            int_sample = [int(float(e if e else 0)) for e in sample]
            helpers.append(IntHelper(int_sample, column_name))
        elif type == dt.DATE:
            logging.debug("Use date helper for '%s'" % column_name)
            date_sample = [datetime.strptime(e, "%Y-%m-%d") for e in sample]
            helpers.append(DateHelper(date_sample, column_name))
        else:
            raise RuntimeError("Unknown type")
    return helpers


def __preprocess(helpers, ndata):
    # Apply the helper's pre-process function on all the values of the corresponding row
    return __apply_helpers(helpers, ndata, lambda h, col: h.preprocess(col))


def __postprocess(helpers, generated_data):
    return [[helpers[column].postprocess(value) for column, value in enumerate(row)] for row in generated_data]


def __apply_helpers(helpers, ndata, helper_func):
    """
    Function that apply a helper operation over the row of a table
    :param helpers: the helper list
    :param ndata: the data to process
    :param helper_func: a function that take a helper and a row of the table and return the processed data
    :return: the processed table
    """
    helper_iter = iter(helpers)
    def apply_helper(col):
        return helper_func(next(helper_iter), col)
    return np.apply_along_axis(apply_helper, 0, ndata)
