import csv
import sdv
import pandas as pd
import os

# The datasets can be found here: https://www.kaggle.com/c/airbnb-recruiting-new-user-bookings

with open("syn_train_users.csv", "w", newline='') as out_file:
    if not os.path.isfile('combined.csv'):
        users = pd.read_csv('train_users.csv')
        countries = pd.read_csv('countries.csv')
        age = pd.read_csv('age_gender_bkts.csv')
        res = pd.merge(users, countries, on='country_destination')
        res = pd.merge(res, age, on='country_destination')
        res.to_csv('combined.csv')

    in_file = open('combined_short.csv', 'r')
    iris_reader = csv.reader(in_file)
    next(iris_reader)  # Skip the header
    ods = [e[1:] for e in iris_reader]

    # The last column (year) is considered categorical since the only value is 2015
    metadata = [sdv.CATEGORICAL, sdv.DATE, sdv.INT, sdv.DATE, sdv.CATEGORICAL, sdv.INT,
                sdv.CATEGORICAL, sdv.INT, sdv.CATEGORICAL, sdv.CATEGORICAL, sdv.CATEGORICAL,
                sdv.CATEGORICAL, sdv.CATEGORICAL, sdv.CATEGORICAL, sdv.CATEGORICAL,
                sdv.CATEGORICAL, sdv.FLOAT, sdv.FLOAT, sdv.FLOAT, sdv.FLOAT, sdv.CATEGORICAL,
                sdv.FLOAT, sdv.CATEGORICAL, sdv.CATEGORICAL, sdv.FLOAT, sdv.CATEGORICAL]
    sds = sdv.syn_by_class(metadata, ods, 15, size=20000)

    iris_writer = csv.writer(out_file)
    iris_writer.writerows(sds)
    in_file.close()
