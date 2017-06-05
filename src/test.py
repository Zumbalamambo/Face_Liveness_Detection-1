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
	# define which model to use
	snapshot_path = "../model/vgg_face_caffe/snapshots"
	model_iter = 20000;
	model_name = '{}/snapshots_iter_{}.caffemodel'.format(snapshot_path, model_iter)

	# load net
	prototxt_path   = "../model/vgg_face_caffe/deploy.prototxt"
	net = caffe.Net(prototxt_path, model_name, caffe.TEST)

	# open test file
	test_image_paths = open("../data/test_images.txt").read().splitlines()
	test_labels = open("../data/test_labels.txt").read().splitlines()
	test_labels = [int(i) for i in test_labels]

	# start infer
	correct_cnt = 0
	total_cnt = 0
	for image, label in zip(test_image_paths, test_labels):
		
		total_cnt += 1
		if total_cnt % 100 == 0:
			acc = float(correct_cnt) / total_cnt
			print "Testing on {}th Image, accuracy is {} ({}/{})".format(total_cnt, acc, correct_cnt, total_cnt)

		im = img_preprocess('../data/' + image)
		net.blobs['data'].data[...] = im
		net.forward()
		pred = net.blobs['fc9_face'].data[0].argmax(axis=0)

		if pred == label:
			correct_cnt += 1

	total_acc = float(correct_cnt) / total_cnt
	print "Total accuracy is {} ({}/{})".format(total_acc, correct_cnt, total_cnt)

