#!/bin/python3

import csv
import os.path
import zipfile

with open('APTMalware/overview.csv','r') as csvfile:
    with open('input/dataset.csv', 'w') as outfile:
        reader = csv.DictReader(csvfile, delimiter=',', quotechar='\'')
        fieldnames = ['Sample', 'is China']
        writer = csv.DictWriter(outfile, fieldnames=fieldnames, delimiter=',', quotechar='\'')
        #writer.writeheader()
        for row in reader:
            if row['Status'] == "X":
                continue
            fname = "APTMalware/samples/{}/{}.zip".format(row['APT-group'],row['SHA256'])
            input_zip = zipfile.ZipFile(str(fname))
            input_zip.setpassword(b"infected")
            samplename = input_zip.namelist()[0]
            input_zip.extract(samplename,path="extracted/")
            fname = "extracted/{}".format(samplename)
            #if not os.path.isfile(fname):
            #    print(fname)
            isChina = 0 if row['Country'] == "China" else 1
            writer.writerow({'Sample': fname, "is China": isChina}) #'Country': row['Country'], 'APT-group': row['APT-group']})
            #print(newrow)
