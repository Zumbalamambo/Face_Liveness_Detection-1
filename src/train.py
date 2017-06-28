import sys
import caffe
import numpy as np
import net
import solver

# init
caffe.set_device(int(sys.argv[1]))
caffe.set_mode_gpu()

# Set paths
dataset_name = "replayattack"
train_prototxt_path = "../model/{}/train.prototxt".format(dataset_name)
val_prototxt_path   = "../model/{}/val.prototxt".format(dataset_name)
solver_path         = "../model/{}/solver.prototxt".format(dataset_name)
snapshot_path       = "../model/{}/snapshots/{}".format(dataset_name, dataset_name)
caffemodel_path     = "../model/vgg_face_caffe/VGG_FACE.caffemodel"

# Create prototxt files
train_batch_size = 10
val_batch_size   = 10
mean = (67.768946, 84.579981, 126.852879)
net.make_net(train_prototxt_path, 'train', train_batch_size, '../data/{}'.format(dataset_name), mean)
net.make_net(val_prototxt_path, 'val', val_batch_size, '../data/{}'.format(dataset_name), mean)
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
