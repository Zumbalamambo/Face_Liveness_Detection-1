import sys
import caffe
import numpy as np
import net
import solver
from PIL import Image

import pdb

import argparse
from util import get_size_dataset, get_size_all, get_mean_dataset

parser = argparse.ArgumentParser()
parser.add_argument('--test_batch_size',    type=int, default=100, help='batch size for testing')
parser.add_argument('--train_dataset_name', type=str, default='myData',  help='name of dataset used for training')
parser.add_argument('--test_dataset_name',  type=str, default='myData',  help='name of dataset to be used for testing')
parser.add_argument('--data_dir',     type=str, default='../data', help='path to the ROOT data folder (disregard the model and dataset used)')
parser.add_argument('--model_name',   type=str, default='VggFace', help='name of model to be used for training')
parser.add_argument('--model_iter',   type=int, default=40000, help='iteration number for the model loaded')
parser.add_argument('--gpu_id',  type=int, default=0,  help='the id of gpu')
opt = parser.parse_args()

# init
caffe.set_device(opt.gpu_id)
caffe.set_mode_gpu()

# define path
# NOTE the difference between (1) path to the test data used, and (2) path to the test prototxt saved (which is same as the train data)
if opt.train_dataset_name == 'all':
	snapshot_path = '../model/{}/snapshots'.format(opt.model_name)
	test_prototxt_path = '../model/{}/test.prototxt'.format(opt.model_name)
else:
	snapshot_path = '../model/{}/{}/snapshots'.format(opt.model_name, opt.train_dataset_name)
	test_prototxt_path = '../model/{}/{}/test.prototxt'.format(opt.model_name, opt.train_dataset_name)	

'''
# make prototxt
if opt.train_dataset_name == 'all':
	train_dataset_mean, train_dataset_size = get_mean_all(opt)
else:
	train_dataset_mean, train_dataset_size = get_mean_dataset(opt)
'''

train_dataset_mean = (142.46179234,   97.86082482,   39.39597104)
net.make_net(test_prototxt_path, 'test', train_dataset_mean, opt)

# load net
model_name = '{}/{}_iter_{}.caffemodel'.format(snapshot_path, opt.train_dataset_name, opt.model_iter)
net = caffe.Net(test_prototxt_path, model_name, caffe.TEST)

# get test dataset information
if opt.test_dataset_name == 'all':
	num_test_samples = get_size_all(opt)
else:
	num_test_samples = get_size_dataset(opt)
niter = int( np.floor( 1.0*num_test_samples/opt.test_batch_size ) )

correct_cnt = 0
total_cnt = 0
labels_all = []
preds_all  = []
for iter in range(niter):

	net.forward()

	preds  = net.blobs['fc9_face'].data.argmax(axis=1)
	labels = np.ndarray.flatten(net.blobs['label'].data)

	preds_all  += preds.tolist()
	labels_all += labels.tolist()

	correct_cnt += np.sum(preds == np.ndarray.flatten(labels))
	total_cnt += opt.test_batch_size
	acc = 1.0 * correct_cnt / total_cnt

	print 'Progress: {}/{}; Accuracy: {} ({}/{})'.format(iter+1, niter, 
													acc, correct_cnt, total_cnt)

total_acc = 1.0 * correct_cnt / total_cnt
print 'Total accuracy is {} ({}/{})'.format(total_acc, correct_cnt, total_cnt)

# compute FPR and FNR
TN = 0
TP = 0
FN = 0
FP = 0
for i in xrange(len(labels_all)):
	if labels_all[i] == 1 and preds_all[i] == 1:
		TP += 1
	if labels_all[i] == 1 and preds_all[i] == 0:
		FN += 1
	if labels_all[i] == 0 and preds_all[i] == 1:
		FP += 1
	if labels_all[i] == 0 and preds_all[i] == 0:
		TN +=1

FPR = 1.0 * FP / (TN + FP)
FNR = 1.0 * FN / (TP + FN)
print 'False Positive Rate (FPR) is {} ({}/{})'.format(FPR, FP, (TN+FP))
print 'False Negative Rate (FNR) is {} ({}/{})'.format(FNR, FN, (TP+FN))
