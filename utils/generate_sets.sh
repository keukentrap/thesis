#!/bin/sh
set +e

cd input
sort --random-sort dataset.csv | head -n 3594 > test.csv
head -n 3100 test.csv > train.csv
tail -n 494 test.csv > predict.csv
