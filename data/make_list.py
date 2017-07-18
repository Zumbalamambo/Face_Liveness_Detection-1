import os
from os import listdir
from os.path import isfile, join
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--name',   type=str, default='myData', help='names of dataset in which the files are to be listed, split by spaces')
parser.add_argument('--concat', action='store_true', help='flag of whether do concatenation to obtain a single large file list')
opt = parser.parse_args()

def get_img_list(dataset_path):
	pos_path  = join(dataset_path, 'cropped', 'positive')
	neg_path  = join(dataset_path, 'cropped', 'negative')
	
	pos_files = [f for f in listdir(pos_path) if isfile(join(pos_path, f))]
	neg_files = [f for f in listdir(neg_path) if isfile(join(neg_path, f))]

	return pos_files, neg_files

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
		pos_images, neg_images = get_img_list(dataset_path)
		pos_labels = [1] * len(pos_images)
		neg_labels = [0] * len(neg_images)
		images_list = pos_images + neg_images
		labels_list = pos_labels + neg_labels 
		if doConcat:
			all_images_list += images_list
			all_labels_list += labels_list

		# write the result into files
		image_list_name = 'image_' + dataset + '.txt'
		with open(image_list_name, 'w') as f:
			for image in images_list:
				f.write('%s\n' % join(cur_path, image))

		label_list_name = 'label_' + dataset + '.txt'
		with open(label_list_name, 'w') as f:
			for label in labels_list:
				f.write('%s\n' % label)

	if doConcat:
		print 'Creating image and label list for all datasets'

		all_image_list_name = 'all_image.txt'
		with open(all_image_list_name, 'w') as f:
			for image in all_images_list:
				f.write('%s\n' % join(cur_path, image))

		all_label_list_name = 'all_label.txt'
		with open(all_label_list_name, 'w') as f:
			for label in all_labels_list:
				f.write('%s\n' % label)