import sys
import caffe
import numpy as np
import net
import solver
from PIL import Image

import pdb

def img_preprocess(im_path, mean=(89.7647, 106.3760, 139.7756)):
	img = Image.open(im_path)
	in_ = np.array(img, dtype=np.float32)
	in_ = in_[:,:,::-1]
	in_ -= np.array(mean)
	in_ = in_.transpose((2,0,1))

	return in_

if __name__ =='__main__':

	# init
	caffe.set_device(int(sys.argv[1]))
	caffe.set_mode_gpu()
	
	# define which model to use
	snapshot_path = "../model/vgg_face_caffe/snapshots"
	model_iter = 30000;
	model_name = '{}/snapshots_iter_{}.caffemodel'.format(snapshot_path, model_iter)

	# create test net
	test_batch_size = 100
	test_prototxt_path = "../model/vgg_face_caffe/test.prototxt"
	net.make_net(test_prototxt_path, 'test', test_batch_size, '../data')

	# load net
	net = caffe.Net(test_prototxt_path, model_name, caffe.TEST)

	# start infer
	num_test_samples = 16511
	niter = int( np.floor( 1.0*num_test_samples/test_batch_size ) )

	correct_cnt = 0
	total_cnt = 0
	for iter in range(niter):

		net.forward()

		preds  = net.blobs['fc9_face'].data.argmax(axis=1)
		labels = net.blobs['label'].data

		correct_cnt += np.sum(preds == np.ndarray.flatten(labels))
		total_cnt += test_batch_size
		acc = 1.0 * correct_cnt / total_cnt

		print "Progress: {}/{}; Accuracy: {} ({}/{})".format(iter+1, niter, 
														acc, correct_cnt, total_cnt)

	total_acc = 1.0 * correct_cnt / total_cnt
	print "Total accuracy is {} ({}/{})".format(total_acc, correct_cnt, total_cnt)