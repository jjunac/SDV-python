import csv
import sdv

with open("categorized_abalone.data", "r", newline='') as in_file, open("categorized_abalone_syn.data", "w", newline='') as out_file:
    abalone_reader = csv.reader(in_file)
    abalone_writer = csv.writer(out_file)

    # Write header
    header = next(abalone_reader)

    ods = [e for e in abalone_reader]

    metadata = [sdv.CATEGORICAL, sdv.FLOAT, sdv.FLOAT, sdv.FLOAT, sdv.FLOAT, sdv.FLOAT, sdv.FLOAT, sdv.FLOAT, sdv.CATEGORICAL]
    sds = sdv.syn(metadata, ods, header=header, size=4177)

    abalone_writer.writerows(sds)
