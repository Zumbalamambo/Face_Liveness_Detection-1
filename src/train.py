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
caffemodel_path     = "../model/vgg_face_caffe/VGG_FACE.caffemodel"
solver_path         = "../model/vgg_face_caffe/solver.prototxt"
snapshot_path       = "../model/vgg_face_caffe/snapshots"

# Create prototxt files
# TODO:
# 1. Shall we tweak learning rate?
# 2. Shall we remove Dropout layer in val.prototxt?\
train_batch_size = 1
val_batch_size   = 1
net.make_net(train_prototxt_path, train_batch_size, val_prototxt_path, val_batch_size)
solver.make_solver(train_prototxt_path, val_prototxt_path, solver_path, snapshot_path)

# Read in solver and pre-trained parameters
mySolver = caffe.get_solver(solver_path)
mySolver.net.copy_from(caffemodel_path)

# Ordinary train loop
mySolver.solve()