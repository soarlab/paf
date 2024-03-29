#!/bin/bash

# Make sure this script is in the home directory of PAF
# This script is going to run PAF on the motivation benchmarks (3 benchmarks)

python3.7 src/main.py -m 53 -e 11 -d 50 -tgc 300 -z -prob 0.999999 benchmarks/CAV2021/MotivationSec/latitude_double.txt
python3.7 src/main.py -m 24 -e 8 -d 50 -tgc 300 -z -prob 0.999999 benchmarks/CAV2021/MotivationSec/latitude_single.txt
python3.7 src/main.py -m 11 -e 5 -d 50 -tgc 300 -z -prob 0.999999 benchmarks/CAV2021/MotivationSec/latitude_half.txt
