import os
from os import listdir
from os.path import isfile, join
import argparse

import pdb

parser = argparse.ArgumentParser()
parser.add_argument('--name',   type=str, default='myData', help='names of dataset in which the files are to be listed, split by spaces')
parser.add_argument('--concat', action='store_true', help='flag of whether do concatenation to obtain a single large file list')
opt = parser.parse_args()

# ge the image list of a certain dataset, given the path to the dataset root
def get_img_list(dataset_path):
	train_pos_path = join(dataset_path, 'train', 'positive')
	train_neg_path = join(dataset_path, 'train', 'negative')
	val_pos_path   = join(dataset_path, 'val', 'positive')
	val_neg_path   = join(dataset_path, 'val', 'negative')
	test_pos_path  = join(dataset_path, 'test', 'positive')
	test_neg_path  = join(dataset_path, 'test', 'negative')

	train_pos_files = [join(train_pos_path, f) for f in listdir(train_pos_path) if isfile(join(train_pos_path, f))]
	train_neg_files = [join(train_neg_path, f) for f in listdir(train_neg_path) if isfile(join(train_neg_path, f))]
	val_pos_files   = [join(val_pos_path, f) for f in listdir(val_pos_path) if isfile(join(val_pos_path, f))]
	val_neg_files   = [join(val_neg_path, f) for f in listdir(val_neg_path) if isfile(join(val_neg_path, f))]
	test_pos_files  = [join(test_pos_path, f) for f in listdir(test_pos_path) if isfile(join(test_pos_path, f))]
	test_neg_files  = [join(test_neg_path, f) for f in listdir(test_neg_path) if isfile(join(test_neg_path, f))]

	print 'positive training samples: %d; negative training samples: %d' % (len(train_pos_files), len(train_neg_files))
	print 'positive validation samples: %d; negative validation samples: %d' % (len(val_pos_files), len(val_neg_files))
	print 'positive testing samples: %d; negative testing samples: %d' % (len(test_pos_files), len(test_neg_files))

	return train_pos_files, train_neg_files, val_pos_files, val_neg_files, test_pos_files, test_neg_files

def write_list_file(dataset_path, prefix, split, image_list, label_list):

	if prefix == 'all':
		image_list_write_path = '{}_{}_image_list.txt'.format(prefix, split)
		with open(image_list_write_path, 'w') as f:
			for image in image_list:
				f.write('%s\n' % image)

		label_list_write_path = '{}_{}_label_list.txt'.format(prefix, split)
		with open(label_list_write_path, 'w') as f:
			for label in label_list:
				f.write('%s\n' % label)	
	else:
		image_list_write_path = '{}/{}_{}_image_list.txt'.format(dataset_path, prefix, split)
		with open(image_list_write_path, 'w') as f:
			for image in image_list:
				f.write('%s\n' % image)

		label_list_write_path = '{}/{}_{}_label_list.txt'.format(dataset_path, prefix, split)
		with open(label_list_write_path, 'w') as f:
			for label in label_list:
				f.write('%s\n' % label)	

if __name__ == '__main__':
	dataset_names = opt.name.split(' ')
	doConcat = opt.concat
	if doConcat:
		all_train_files  = []
		all_train_labels = []
		all_test_files   = []
		all_test_labels  = []
		all_val_files    = []
		all_val_labels   = []

	cur_path = os.path.dirname(os.path.abspath(__file__))
	for dataset in dataset_names:

		print '====================================================================='
		print 'Creating image and label list for %s' % dataset
		dataset_path = join(cur_path, dataset)

		# get the file list and create the label list (only file names are obtained)
		train_pos_files, train_neg_files, val_pos_files, val_neg_files, test_pos_files, test_neg_files = get_img_list(dataset_path)

		train_files = train_pos_files + train_neg_files
		val_files   = val_pos_files   + val_neg_files
		test_files  = test_pos_files  + test_neg_files	

		train_labels = [1] * len(train_pos_files) + [0] * len(train_neg_files)
		val_labels   = [1] * len(val_pos_files)   + [0] * len(val_neg_files)
		test_labels  = [1] * len(test_pos_files)  + [0] * len(test_neg_files)

		# write the result into files
		write_list_file(dataset_path, dataset, 'train', train_files, train_labels)
		write_list_file(dataset_path, dataset, 'val',   val_files,   val_labels)
		write_list_file(dataset_path, dataset, 'test',  test_files,  test_labels)

		if doConcat:
			all_train_files  += train_files
			all_train_labels += train_labels
			all_val_files    += val_files
			all_val_labels   += val_labels
			all_test_files   += test_files
			all_test_labels  += test_labels

	if doConcat:
		# write the result into files
		print '====================================================================='
		print 'Creating image and label list for all datasets'
		write_list_file(cur_path, 'all', 'train', all_train_files, all_train_labels)
		write_list_file(cur_path, 'all', 'val',   all_val_files,   all_val_labels)
		write_list_file(cur_path, 'all', 'test',  all_test_files,  all_test_labels)