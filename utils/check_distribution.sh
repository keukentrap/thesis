#! /bin/bash

for i in $(seq 0 4);
do
	grep ",$i" -c ../input/train.csv ../input/predict.csv
	echo ""
done
