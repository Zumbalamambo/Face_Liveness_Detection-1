import os
from PIL import Image
import colorsys
import numpy as np
'''
For training dataset, return both mean and size
'''
def get_mean_dataset(opt):
	# we only care about the mean of training images
	train_list_file = os.path.join(opt.data_dir, opt.train_dataset_name, opt.train_dataset_name+'_train_image_list.txt')
	train_images = open(train_list_file).read().splitlines()
	
	mean_value = np.array([0, 0, 0], dtype=np.float64)
	cnt = 0
	for image in train_images:
		im = np.asarray(Image.open(image))
		mean_value += np.mean(np.mean(im, axis=0), axis=0)
		cnt += 1

	mean_value = mean_value/cnt

	# RGB -> BGR
	mean_value = mean_value[::-1]

	print 'mean value for %s is (%f, %f, %f) ' % (opt.train_dataset_name, mean_value[0], mean_value[1], mean_value[2])
	print 'total number of training samples are %d' % cnt
	return mean_value, cnt

def get_mean_all(opt):
	total_sum = 0
	total_cnt = 0
	dataset_names = os.listdir(opt.data_dir)
	for dataset in dataset_names:
		if os.path.isdir(os.path.join(opt.data_dir, dataset)):
			mean_value, cnt = get_mean_dataset(opt.data_dir, dataset)
			total_sum += mean_value * cnt
			total_cnt += cnt

	total_mean = total_sum/total_cnt
	print 'mean value for all dataset is (%f, %f, %f), total number of training samples are %d' % (total_mean[0], total_mean[1], total_mean[2], total_cnt)
	return total_mean, total_cnt

'''
For testing dataset, return only sizes
'''
def get_size_dataset(opt):
	test_list_file = os.path.join(opt.data_dir, opt.test_dataset_name, opt.test_dataset_name+'_test_image_list.txt')
	with open(test_list_file) as f:
		return len(f.readlines())

def get_size_all(opt):
	total = 0
	dataset_names = os.listdir(opt.data_dir)
	for dataset in dataset_names:
		total += get_size_dataset(opt)
	return total

'''
convert RGB to HSV
'''
def HSVColor(img):
    if isinstance(img, Image.Image):
        r,g,b = img.split()
        Hdat = []
        Sdat = []
        Vdat = [] 
        for rd,gn,bl in zip(r.getdata(),g.getdata(),b.getdata()) :
            h,s,v = colorsys.rgb_to_hsv(rd/255.,gn/255.,bl/255.)
            Hdat.append(int(h*255.))
            Sdat.append(int(s*255.))
            Vdat.append(int(v*255.))
        r.putdata(Hdat)
        g.putdata(Sdat)
        b.putdata(Vdat)
        return Image.merge('RGB',(r,g,b))
    else:
        return None