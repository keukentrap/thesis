#!/bin/python3

##
# Used to create the .csv file meant for training
##

import csv
import os.path
import zipfile

def label_by(row, name, labels):
    if not row[name] in labels:
        if len(labels) == 0:
            labels[row[name]] = 0
        else:
            labels[row[name]] = max(labels.values()) + 1

    c = labels[row[name]]
    #isChina = 0 if row['Country'] == "China" else 1
    return c


with open('APTMalware/overview.csv','r') as csvfile:
    with open('../input/dataset.csv', 'w') as outfile:
        reader = csv.DictReader(csvfile, delimiter=',', quotechar='\'')
        fieldnames = ['Sample', 'Label']
        writer = csv.DictWriter(outfile, fieldnames=fieldnames, delimiter=',', quotechar='\'')
        labels = {}
        #writer.writeheader()
        for row in reader:
            if row['Status'] == "X":
                continue
            fname = "APTMalware/samples/{}/{}.zip".format(row['APT-group'],row['SHA256'])
            input_zip = zipfile.ZipFile(str(fname))
            input_zip.setpassword(b"infected")
            samplename = input_zip.namelist()[0]
            #input_zip.extract(samplename,path="extracted/")
            fname = "../dataset/extracted/{}".format(samplename)
            #if not os.path.isfile(fname):
            #    print(fname)

            # Per country:
            c = label_by(row,'Country',labels)

            # China vs North-Korea
            #if not c in [0,2]:
            #    continue

            #if c == 2:
            #    c = 1


            # Per apt-group
            #c = label_by(row,'APT-group',labels)

            writer.writerow({'Sample': fname, "Label": c }) #'Country': row['Country'], 'APT-group': row['APT-group']})
