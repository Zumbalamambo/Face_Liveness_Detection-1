import sys
import caffe
import numpy as np
import net
import solver

# init
caffe.set_device(int(sys.argv[1]))
caffe.set_mode_gpu()

# Set paths
train_prototxt_path = "../model/vgg_face_caffe/train.prototxt"
val_prototxt_path   = "../model/vgg_face_caffe/val.prototxt"
test_prototxt_path  = "../model/vgg_face_caffe/test.prototxt"
caffemodel_path     = "../model/vgg_face_caffe/VGG_FACE.caffemodel"
solver_path         = "../model/vgg_face_caffe/solver.prototxt"
snapshot_path       = "../model/vgg_face_caffe/snapshots/model"

# Create prototxt files
# TODO:
# 1. Shall we tweak learning rate?
# 2. Shall we remove Dropout layer in val.prototxt? No
train_batch_size = 10
val_batch_size   = 10
net.make_net(train_prototxt_path, 'train', train_batch_size, '../data')
net.make_net(val_prototxt_path, 'val', val_batch_size, , '../data')
solver.make_solver(train_prototxt_path, val_prototxt_path, solver_path, snapshot_path)

# Read in solver and pre-trained parameters
mySolver = caffe.get_solver(solver_path)
mySolver.net.copy_from(caffemodel_path)

# Ordinary train loop
test_interval = 500
train_loss = np.zeros(mySolver.param.max_iter)
val_acc    = np.zeros( int( np.ceil( mySolver.param.max_iter/test_interval ) ) )

for iter in range(mySolver.param.max_iter):
	mySolver.step(1)

	# store the training loss
	train_loss[iter] = mySolver.net.blobs['loss'].data

	# store the validation accuracy
	if iter % test_interval == 0:
		acc = mySolver.test_nets[0].blobs['acc'].data
		val_acc[iter // test_interval] = acc
		print '=== Validation Accuracy for iter = {}: {} ==='.format(iter, acc)
