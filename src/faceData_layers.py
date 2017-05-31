import caffe

import numpy as np
from PIL import Image
import copy

class FaceDataLayer(caffe.Layer):
	"""
	Load (face image, face label) pairs from created dataset
	The input size is fixed to be batchsize x 3 x 224 x 224
	"""

	def setup(self, bottom, top):
		"""
		Setup data layer according to parameters:

		- data_dir:  path to image file list (a list of paths)
		- label_dir: path to label list (a list of labels)
		- split:     train / val / test
		- mean:      tuple of mean values to subtract
		- batchsize: size of batch		

		"""

		# start config
		params = eval(self.param_str) # do param_str in python console (create dict)
		self.data_dir = params['data_dir']
		self.label_dir = params['label_dir'] 
		self.split = params['split']
		self.mean = np.array(params['mean'])
		self.batch_size = params['batch_size']
		
		# load image paths and their labels (both are list of strings)
		self.image_paths = open(self.data_dir).read().splitlines()
		self.labels = open(self.label_dir).read().splitlines()
		self.num_samples = len(image_paths)		

		# set the dataset offset to 0
		self.offset = 0

	def reshape(self, bottom, top):
		# called before every forward(), load image and label to net
		# reshape two top blobs to define data shape. For details, see:
		# https://github.com/BVLC/caffe/blob/master/examples/pycaffe/layers/pascal_multilabel_datalayers.py

	def forward(self, bottom, top):
		# assign output for top blob
		top[0].data[...] = self.data
		top[1].data[...] = self.label

		# pick next batch, shuffle if necessary
		# TODO		

	def backward(self, top, propagate_down, bottome):
		pass

	def shuffle_dataset(self):
		# shuffle the dataset

		# shuffle the image path and label using the same permutation
		permute_idx = np.random.permutation(self.num_samples)
		image_paths_tmp = [self.image_paths[i] for i in permute_idx]
		labels_tmp = [self.labels[i] for i in permute_idx]
		
		self.image_paths = copy.copy(image_paths_tmp)
		self.labels = copy.copy(labels_tmp)

		# reset the offset
		self.offset = 0

	def get_next_batch(self):
		# return the next batch of (image, label) pairs
		start_offset = self.offset
		self.offset += self.batch_size

		if self.offset > self.num_samples:
			# current epoch finished
			self.shuffle_dataset()
			start_offset = 0
			self.offset = batch_size
		
		end_offset = self.offset

		
