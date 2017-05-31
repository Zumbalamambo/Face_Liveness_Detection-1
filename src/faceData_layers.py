import caffe

import numpy as np
from PIL import Image

import random

class FaceDataLayer(caffe.Layer):
	"""
	Load (face image, face label) pairs from created dataset
	The input size is fixed to be 1 x 3 x 224 x 224
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
		self.batchsize = params['batchsize']
		
		# blob validation
		if len(top) != 2:
			raise Exception("Need to define two tops: data and label.")
		if len(bottom) != 0:
			raise Exception("Do not define a bottom.")

		# load paths for data and label
		self.image_paths = open(self.data_dir).read().splitlines()
		self.labels = open(self.label_dir).read().splitlines()
		
		# set base index
		self.idx = 0

	def reshape(self, bottom, top)
