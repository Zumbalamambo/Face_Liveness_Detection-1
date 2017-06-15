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
	
	# define which model to use
	snapshot_path = '../model/vgg_face_caffe/snapshots'
	model_iter = 30000;
	model_name = '{}/snapshots_iter_{}.caffemodel'.format(snapshot_path, model_iter)

	# create test net
	test_batch_size = 100
	test_prototxt_path = '../model/vgg_face_caffe/test.prototxt'
	data_dir = '../data/3rd_party/MSU-MFSD'
	mean = (110.3397, 125.8261, 155.2444)
	net.make_net(test_prototxt_path, 'test', test_batch_size, data_dir, mean)

	# load net
	net = caffe.Net(test_prototxt_path, model_name, caffe.TEST)

	# start testing
	num_test_samples = 61302
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