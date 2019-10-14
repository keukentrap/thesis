#! /bin/bash

for i in $(seq 1 11);
do
	grep ",$i" -c input/train.csv input/predict.csv
	echo ""
done
