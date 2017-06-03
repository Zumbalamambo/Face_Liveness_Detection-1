import caffe

import numpy as np
from PIL import Image
import copy

"""
Reference: 
https://github.com/BVLC/caffe/blob/master/examples/pycaffe/layers/pascal_multilabel_datalayers.py
https://github.com/yunfan0621/fcn.berkeleyvision.org/blob/master/voc_layers.py
"""

class FaceDataLayer(caffe.Layer):
	"""
	Load (face image, face label) pairs from created dataset
	The input image size is fixed to be 3 x 224 x 224 with 2 labels
	"""

	def setup(self, bottom, top):

		self.top_names = ['data', 'label']

		# start config
		params = eval(self.param_str) # do param_str in python console (create dict)

		self.batch_size = params['batch_size']
		self.batch_loader = BatchLoader(params, None) # save all other params to loader 

		"""
		since we use a fixed input image size, reshape the data layer only once
		and save it from be called every time in reshape()
		"""
		top[0].reshape(self.batch_size, 3, 224, 224)
		top[1].reshape(self.batch_size, 1)

	def forward(self, bottom, top):
		# assign output for top blob
		for iter in range(self.batch_size):
			# load (image, label) pair via batch loader
			# NOTE: only one single pair is loaded at a time (not a batch)
			im, label = self.batch_loader.load_next_pair()

			# assign data and label to data layer
			top[0].data[iter, ...] = im
			top[1].data[iter, ...] = label

	def backward(self, top, propagate_down, bottom):
		pass

	def reshape(self, bottom, top):
		# reshaping done in layer setup
		pass


class BatchLoader(object):
	
	"""
    This class abstracts away the loading of images (for the ease of debugging)
    Images are loaded individually.
    """

	def __init__(self, params, result):
			# load image paths and their labels (both are list of strings)

		self.data_dir = params['data_dir']
		self.split = params['split']
		self.image_paths = open('{}/{}_images.txt'.format(self.data_dir, self.split)).read().splitlines()
		self.labels      = open('{}/{}_labels.txt'.format(self.data_dir, self.split)).read().splitlines()
		
		self.batch_size = params['batch_size']
		self.mean = np.array(params['mean'])
		self._cur = 0 # index of current image

	def load_next_pair(self):
		# return the next batch of (image, label) pairs

		# check whether an epoch has been finished
		if self._cur == len(self.image_paths):
			self._cur = 0
			self.shuffle_dataset()

		im = np.asarray(Image.open('{}/{}'.format(self.data_dir, self.image_paths[self._cur])))
		label = int(self.labels[self._cur])

		# do a simple horizontal flip as data augmentation
		flip = np.random.choice(2)*2-1
		im = im[:, ::flip, :]

		self._cur += 1
		return self.preprocessor(im), label

	def shuffle_dataset(self):
		# shuffle the dataset

		# shuffle the image path and label using the same permutation
		permute_idx = np.random.permutation(len(self.image_paths))
		image_paths_tmp = [self.image_paths[i] for i in permute_idx]
		labels_tmp = [self.labels[i] for i in permute_idx]
		
		self.image_paths = copy.copy(image_paths_tmp)
		self.labels = copy.copy(labels_tmp)

		# reset the offset
		self._cur = 0

	def preprocessor(self, im):
		"""
		preprocess the image for caffe use:
		- cast to float
		- switch channels RGB -> BGR
		- subtract mean
		- transpose to channel x height x width order
		"""

		im = np.array(im, dtype=np.float32)
		im = im[:,:,::-1] # RGB -> BGR
		im -= self.mean # the order of mean should be BGR
		im = im.transpose((2, 0, 1))
		return im