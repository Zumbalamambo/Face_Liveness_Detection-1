import sys
import caffe
import numpy as np
import net
import solver
from PIL import Image

import pdb

if __name__ =='__main__':

	# init
	caffe.set_device(int(sys.argv[1]))
	caffe.set_mode_gpu()
	
	train_dataset_name = 'replayattack' 
	test_dataset_name  = 'cbsr_antispoofing' 
	# load trained net
	snapshot_path = '../model/{}/snapshots'.format(train_dataset_name)
	model_iter = 30000;
	model_name = '{}/{}_iter_{}.caffemodel'.format(snapshot_path, train_dataset_name, model_iter)

	# create test net
	test_batch_size = 100
	test_prototxt_path = '../model/{}/test.prototxt'.format(test_dataset_name)
	data_dir = '../data/{}'.format(test_dataset_name)
	'''
	mean value for trained model:
		myData: (89.7647, 106.376, 139.7756)
		MSU-MFSD: (110.266284, 125.741553, 155.540794)
		cbsr_antispoofing: (93.976333, 101.775137, 126.129148)
		replayattack: (67.768946, 84.579981, 126.852879)
	'''
	mean = (67.768946, 84.579981, 126.852879)
	net.make_net(test_prototxt_path, 'test_all', test_batch_size, data_dir, mean)

	# load net
	net = caffe.Net(test_prototxt_path, model_name, caffe.TEST)

	'''
	num_test_samples for each dataset:
		test all:
			cbsr_antispoofing: 51754
			MSU-MFSD: 61302
			replayattack: 95994
			myData: 82552
		test:
			myData: 16511
			MSU-MFSD: 15463
			cbsr_antispoofing: 10351
			replayattack: 19199
	'''
	num_test_samples = 51754
	niter = int( np.floor( 1.0*num_test_samples/test_batch_size ) )

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
		total_cnt += test_batch_size
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
