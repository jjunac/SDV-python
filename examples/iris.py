import csv
import sdv

with open("iris.data", "r", newline='') as in_file, open("synthetic_iris3.data", "w", newline='') as out_file:
    iris_reader = csv.reader(in_file)
    ods = [e for e in iris_reader]

    metadata = [sdv.FLOAT, sdv.FLOAT, sdv.FLOAT, sdv.FLOAT, sdv.CATEGORICAL]
    sds = sdv.syn(metadata, ods, size=150)

    iris_writer = csv.writer(out_file)
    iris_writer.writerows(sds)