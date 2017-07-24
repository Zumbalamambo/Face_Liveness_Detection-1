#!/bin/bash

set -x
set -e

LOG="../log/train_VggFace_all.txt.`date +'%Y-%m-%d_%H-%M-%S'`"
exec &> >(tee -a "$LOG")
echo Logging output to "$LOG"

time python ../src/train.py --dataset_name 'all' \
                            --model_name 'VggFace'
