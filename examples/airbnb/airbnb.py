import csv
import sdv
import pandas as pd
import os

# The datasets can be found here: https://www.kaggle.com/c/airbnb-recruiting-new-user-bookings

with open("syn_train_users.csv", "w", newline='') as out_file:
    in_file = open('combined_10000.csv', 'r')
    iris_reader = csv.reader(in_file)
    next(iris_reader)  # Skip the header
    ods = [e for e in iris_reader]

    # The last column (year) is considered categorical since the only value is 2015
    metadata = [sdv.CATEGORICAL, sdv.DATE, sdv.INT, sdv.DATE, sdv.CATEGORICAL, sdv.INT,
                sdv.CATEGORICAL, sdv.INT, sdv.CATEGORICAL, sdv.CATEGORICAL, sdv.CATEGORICAL,
                sdv.CATEGORICAL, sdv.CATEGORICAL, sdv.CATEGORICAL, sdv.CATEGORICAL,
                sdv.CATEGORICAL, sdv.FLOAT, sdv.FLOAT, sdv.FLOAT, sdv.FLOAT, sdv.CATEGORICAL,
                sdv.FLOAT, sdv.CATEGORICAL, sdv.CATEGORICAL, sdv.FLOAT, sdv.CATEGORICAL]
    sds = sdv.syn(metadata, ods, size=10000)

    iris_writer = csv.writer(out_file)
    iris_writer.writerows(sds)
    in_file.close()
