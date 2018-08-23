import csv
import sdv

with open("categorized_abalone.data", "r", newline='') as in_file, open("categorized_abalone_syn.data", "w", newline='') as out_file:
    abalone_reader = csv.reader(in_file)

    header = next(abalone_reader)

    ods = [e for e in abalone_reader]

    metadata = [sdv.CATEGORICAL, sdv.FLOAT, sdv.FLOAT, sdv.FLOAT, sdv.FLOAT, sdv.FLOAT, sdv.FLOAT, sdv.FLOAT, sdv.CATEGORICAL]
    sds = sdv.syn_by_class(metadata, ods, 8, size=4177)

    abalone_writer = csv.writer(out_file)
    abalone_writer.writerow(header)
    abalone_writer.writerows(sds)
