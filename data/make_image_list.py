import os
from os import listdir
from os.path import isfile, join
import argparse
from random import shuffle

import pdb

parser = argparse.ArgumentParser()
parser.add_argument('--name',   type=str, default='myData', help='names of dataset in which the files are to be listed, split by spaces')
parser.add_argument('--concat', action='store_true', help='flag of whether do concatenation to obtain a single large file list')
opt = parser.parse_args()

def get_img_list(dataset_path):
	pos_path  = join(dataset_path, 'cropped', 'positive')
	neg_path  = join(dataset_path, 'cropped', 'negative')
	
	pos_files = [join(pos_path, f) for f in listdir(pos_path) if isfile(join(pos_path, f))]
	neg_files = [join(pos_path, f) for f in listdir(neg_path) if isfile(join(neg_path, f))]

	return pos_files, neg_files

def make_split_list(images, labels, save_path, prefix):
	indices = list(range(len(images)))
	shuffle(indices)
	images = [images[i] for i in indices]
	labels = [labels[i] for i in indices]

	# create the list of each split
	ntotal = len(images)
	ntest  = int(ntotal * 0.2)
	nval   = int((ntotal - ntest) * 0.2)
	ntrain = ntotal - ntest - nval

	train_images = images[0 : ntrain]
	train_labels = labels[0 : ntrain]
	val_images = images[ntrain : ntrain + nval]
	val_labels = labels[ntrain : ntrain + nval]
	test_images = images[ntrain + nval : ntrain + nval + ntest]
	test_labels = labels[ntrain + nval : ntrain + nval + ntest]

	# write the result into files
	with open(join(save_path, prefix+'_train_image_list.txt'), 'w') as f:
		for image in train_images:
			f.write('%s\n' % image)

	with open(join(save_path, prefix+'_train_label_list.txt'), 'w') as f:
		for label in train_labels:
			f.write('%s\n' % label)

	with open(join(save_path, prefix+'_val_image_list.txt'), 'w') as f:
		for image in val_images:
			f.write('%s\n' % image)

	with open(join(save_path, prefix+'_val_label_list.txt'), 'w') as f:
		for label in val_labels:
			f.write('%s\n' % label)

	with open(join(save_path, prefix+'_test_image_list.txt'), 'w') as f:
		for image in test_images:
			f.write('%s\n' % image)

	with open(join(save_path, prefix+'_test_label_list.txt'), 'w') as f:
		for label in test_labels:
			f.write('%s\n' % label)

if __name__ == '__main__':
	dataset_names = opt.name.split(' ')
	doConcat = opt.concat
	if doConcat:
		all_images_list = []
		all_labels_list = []

	cur_path = os.path.dirname(os.path.abspath(__file__))
	for dataset in dataset_names:

		print 'Creating image and label list for %s' % dataset
		dataset_path = join(cur_path, dataset)

		# get the file list and create the label list 
		pos_images, neg_images = get_img_list(dataset_path) # absolute paths
		pos_labels = [1] * len(pos_images)
		neg_labels = [0] * len(neg_images)
		images_list = pos_images + neg_images
		labels_list = pos_labels + neg_labels
		if doConcat:
			all_images_list += images_list
			all_labels_list += labels_list

		make_split_list(images_list, labels_list, dataset_path, dataset)

	if doConcat:
		print 'Creating image and label list for all datasets'
		make_split_list(all_images_list, all_labels_list, cur_path, 'all')