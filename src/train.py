import os
import sys
import caffe
import net
import solver

import numpy as np
import argparse
import time
import pdb

from util import get_mean_all, get_mean_dataset

parser = argparse.ArgumentParser()
parser.add_argument('--train_batch_size', type=int, default=20,  help='batch size for training')
parser.add_argument('--val_batch_size',   type=int, default=50, help='batch size for validation')
parser.add_argument('--num_epoch',     type=int, default=5, help='number of epochs')
parser.add_argument('--test_interval', type=int, default=200, help='interval for invoking validation/testing')
parser.add_argument('--data_dir',      type=str, default='../data', help='path to the data folder')
parser.add_argument('--model_name',    type=str, default='VggFace', help='name of model to be used for training')
parser.add_argument('--train_dataset_name', type=str, default='MZDX', help='name of dataset to be used for training')
parser.add_argument('--use_HSV',  action='store_true', help='indicator of using HSV colorspace (set to use)')
parser.add_argument('--gpu_id',  type=int, default=0,  help='the id of gpu')
opt = parser.parse_args()

# init
caffe.set_device(opt.gpu_id)
caffe.set_mode_gpu()

# set paths
model_path = os.path.join('..', 'model', opt.model_name)
if not os.path.exists(model_path):
	os.makedirs(model_path)

caffemodel_path = "{}/{}_model/{}.caffemodel".format(model_path, opt.model_name, opt.model_name)
if opt.train_dataset_name == 'all':
	# use all the datasets (all folders have already been makde)
	train_prototxt_path = "{}/train.prototxt".format(model_path)
	val_prototxt_path   = "{}/val.prototxt".format(model_path)
	solver_path         = "{}/solver.prototxt".format(model_path)
	snapshot_path       = "{}/snapshots/all".format(model_path)	
else:	
	# use a specific dataset (make folder for the dataset model and snapshots)
	model_dataset_path = os.path.join(model_path, opt.train_dataset_name)
	if not os.path.exists(model_dataset_path):
		os.makedirs(model_dataset_path)

	model_dataset_snapshots_path = os.path.join(model_dataset_path, 'snapshots')
	if not os.path.exists(model_dataset_snapshots_path):
		os.makedirs(model_dataset_snapshots_path)

	train_prototxt_path = "{}/train.prototxt".format(model_dataset_snapshots_path)
	val_prototxt_path   = "{}/val.prototxt".format(model_dataset_snapshots_path)
	solver_path         = "{}/solver.prototxt".format(model_dataset_snapshots_path)
	snapshot_path       = "{}/snapshots/{}".format(model_dataset_snapshots_path, opt.train_dataset_name)

'''
if opt.train_dataset_name == 'all':
	dataset_mean, dataset_size = get_mean_all(opt)
else:
	dataset_mean, dataset_size = get_mean_dataset(opt)
'''
# MZDX:     dataset_mean = (94.472229, 105.928444, 135.969991)  dataset_size = 164290
# MZDX_HSV: dataset_mean = (142.461792, 97.860825, 39.395971)   dataset_size = 164290

# dataset_mean = (142.461792, 97.860825, 39.395971)
# dataset_size = 164293

# make prototxt
net.make_net(train_prototxt_path, 'train',  dataset_mean, opt)
net.make_net(val_prototxt_path,   'val',    dataset_mean, opt)
solver.make_solver(train_prototxt_path, val_prototxt_path, 
				   solver_path, snapshot_path, 
				   opt, dataset_size, opt.test_interval)

# Read in solver and pre-trained parameters
mySolver = caffe.get_solver(solver_path)
mySolver.net.copy_from(caffemodel_path)

# Ordinary train loop
train_loss = np.zeros(mySolver.param.max_iter)
val_acc    = np.zeros(int(np.ceil(mySolver.param.max_iter/test_interval)) + 1)

print 'Total number of iterations: {}'.format(mySolver.param.max_iter)
for iter in range(mySolver.param.max_iter):
	mySolver.step(1)

	# store the training loss
	train_loss[iter] = mySolver.net.blobs['loss'].data

	# store the validation accuracy
	if iter % test_interval == 0:
		acc = mySolver.test_nets[0].blobs['acc'].data
		val_acc[iter // test_interval] = acc
		print '=== Validation Accuracy for iter = {}: {} ==='.format(iter, acc)

# log the train error and val acc
localtime  = time.localtime()
timeString = time.strftime("%Y_%m_%d_%H_%M_%S", localtime)

train_error_save_path = '../results/%s_train_error_' % opt.train_dataset_name + timeString + '.txt'
with open (train_error_save_path, 'w') as f:
	for loss in train_loss:
		f.write('%s\n' % str(loss)) 

val_acc_save_path =  '../results/%s_val_acc_' % opt.train_dataset_name + timeString + '.txt'
with open (val_acc_save_path, 'w') as f:
	for acc in val_acc:
		f.write('%s\n' % str(acc)) 