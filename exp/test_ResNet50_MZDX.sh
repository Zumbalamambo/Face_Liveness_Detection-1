!/bin/bash
set -x
set -e

FILENAME=`basename $0`
LOG="../log/$FILENAME.txt.`date +'%Y-%m-%d_%H-%M-%S'`"
exec &> >(tee -a "$LOG")
echo Logging output to "$LOG"

time python ../src/test.py --test_batch_size 20 \
			   --train_dataset_name 'MZDX' \
                           --test_dataset_name  'MZDX' \
                           --model_name 'ResNet50' \
                           --model_iter 5000 \
