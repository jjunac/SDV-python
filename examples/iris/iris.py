import csv
import sdv

with open("iris.data", "r", newline='') as in_file, open("iris_syn.data", "w", newline='') as out_file:
    iris_reader = csv.reader(in_file)
    header = next(iris_reader)
    ods = [e for e in iris_reader]

    metadata = [sdv.FLOAT, sdv.FLOAT, sdv.FLOAT, sdv.FLOAT, sdv.CATEGORICAL]
    sds = sdv.syn_by_class(metadata, ods, 4, header=header, size=150)

    iris_writer = csv.writer(out_file)
    iris_writer.writerow(header)
    iris_writer.writerows(sds)
