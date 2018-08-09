import csv
import sdv

with open("abalone.data", "r", newline='') as in_file, open("synthetic_abalone.data", "w", newline='') as out_file:
    abalone_reader = csv.reader(in_file)
    ods = [e for e in abalone_reader]

    metadata = [sdv.CATEGORICAL, sdv.FLOAT, sdv.FLOAT, sdv.FLOAT, sdv.FLOAT, sdv.FLOAT, sdv.FLOAT, sdv.FLOAT, sdv.INT]
    sds = sdv.syn(metadata, ods, size=4177)

    abalone_writer = csv.writer(out_file)
    abalone_writer.writerows(sds)
