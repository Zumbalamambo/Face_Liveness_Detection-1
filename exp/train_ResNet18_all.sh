#!/bin/bash

set -x
set -e

FILENAME=`basename $0`
LOG="../log/$FILENAME.txt.`date +'%Y-%m-%d_%H-%M-%S'`"
exec &> >(tee -a "$LOG")
echo Logging output to "$LOG"

time python ../src/train.py     --base_lr 1e-5 \
                                --lr_policy 'step'\
                                --stepsize 5000 \
                                --train_batch_size 64 \
                                --val_batch_size 64\
                                --test_interval 200 \
                                --num_epoch 5\
                                --train_dataset_name 'all' \
                                --model_name 'ResNet18' \
                                --gpu_id 3
