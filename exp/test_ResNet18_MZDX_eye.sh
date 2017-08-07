#!/bin/bash

set -x
set -e

FILENAME=`basename $0`
LOG="../log/$FILENAME.txt.`date +'%Y-%m-%d_%H-%M-%S'`"
exec &> >(tee -a "$LOG")
echo Logging output to "$LOG"

time python ../src/test.py      --train_dataset_name 'MZDX_eye' \
                                --test_dataset_name 'MZDX_eye' \
                                --test_batch_size 60 \
                                --model_name 'ResNet18' \
                                --model_iter 25000 \
                                --gpu_id 2
