from caffe.proto import caffe_pb2

def make_solver(train_net_path, val_net_path, solver_path, snapshot_path):
	s = caffe_pb2.SolverParameter()

	# specify locations of the train and test networks.
	s.train_net = train_net_path
	s.test_net.append(val_net_path)

	# specify parameters for iterations
	s.test_interval = 500 # interval for invoking testing
	s.test_iter.append(20) # number of batches used for testing
	s.max_iter = 100000

	# specify parameters for learning policy
	s.base_lr = 1e-8
	s.lr_policy = "step"
	s.gamma = 0.1

	s.type = "Adam"
	s.momentum = 0.9
	s.weight_decay= 5e-4
	s.iter_size = 1 # no gradient accumulation

	# specify other helper parameters
	s.display = 20
	s.snapshot = 5000
	s.snapshot_prefix = snapshot_path
	s.solver_mode = caffe_pb2.SolverParameter.GPU

	print "Writing prototxt file for solver..."
	with open(solver_path, 'w') as f:
		f.write(str(s))

