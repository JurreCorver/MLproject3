#!/bin/bash

for ((nBoost = 100; nBoost < 1001; nBoost = nBoost + 100));do

    bsub -n 1 "adaLogisRegEuler.py $((nBoost))"

done;