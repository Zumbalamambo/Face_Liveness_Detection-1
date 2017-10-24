#!/bin/bash

set -x
set -e

FILENAME=`basename $0`
LOG="../log/$FILENAME.txt.`date +'%Y-%m-%d_%H-%M-%S'`"
exec &> >(tee -a "$LOG")
echo Logging output to "$LOG"

time python ../src/test.py      --load_size 256 \
                                --crop_size 227 \
                                --train_dataset_name 'all' \
                                --test_dataset_name 'all' \
                                --test_batch_size 144 \
                                --model_name 'SqueezeNet' \
                                --model_iter 12500 \
                                --exp_suffix 'lowLR' \
                                --gpu_id 4
