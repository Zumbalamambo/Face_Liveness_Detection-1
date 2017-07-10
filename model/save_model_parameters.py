import sys
import caffe
import numpy as np
import collections
import pickle

import pdb

# init
caffe.set_device(int(sys.argv[1]))
caffe.set_mode_gpu()

# Set paths and load the parameters
prototxt_path   = "./vgg_face_caffe/deploy.prototxt" 
caffemodel_path = "./vgg_face_caffe/VGG_FACE.caffemodel"
net = caffe.Net(prototxt_path, caffemodel_path, caffe.TEST)

# extract the layer parameters
vgg_face_weights = collections.OrderedDict()
vgg_face_bias    = collections.OrderedDict()
for layer_name, param in net.params.iteritems():
	vgg_face_weights[layer_name] = param[0].data
	vgg_face_bias[layer_name] = param[1].data

# save the retrieved weights and bias
weights_path = "./vgg_face_caffe/weights.pkl"
bias_path = "./vgg_face_caffe/bias.pkl"
with open(weights_path, 'w') as f:
	pickle.dump(vgg_face_weights, f)
with open(bias_path, 'w') as f:
	pickle.dump(vgg_face_bias, f)