#!/bin/bash

set -x
set -e

FILENAME=`basename $0`
LOG="../log/$FILENAME.txt.`date +'%Y-%m-%d_%H-%M-%S'`"
exec &> >(tee -a "$LOG")
echo Logging output to "$LOG"

time python ../src/train.py  --base_lr 1e-6 \
                                --lr_policy 'step'\
                                --stepsize 5000 \
                                --train_batch_size 20 \
                                --test_interval 200 \
                                --num_epoch 10\
                                --train_dataset_name 'MZDX' \
                                --model_name 'VggFace' \
