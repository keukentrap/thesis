#!/bin/sh
set +e

cd ../input
# 3594 samples
shuf dataset.csv -o test.csv

head -n 3200 test.csv > train.csv
tail -n 394 test.csv > predict.csv

# China vs. North Korea
# 1611 samples
#head -n 1400 test.csv > train.csv
#tail -n 211 test.csv > predict.csv
