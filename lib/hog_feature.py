# -*- coding: utf-8 -*-
import cv2, os, sys
sys.path.append('/Users/pengfeiwang/Documents/anaconda2/lib/python2.7/site-packages')
import pandas as pd

pic_path = '/Users/pengfeiwang/Desktop/dogkfc/Project3_poodleKFC_train/images/'
winSize = (64, 64)
block_size= (16, 16)
block_stride= (8, 8)
cell_size= (8, 8)
nbins = 9
hog = cv2.HOGDescriptor(winSize, block_size, block_stride, cell_size, nbins)

padding = (8,8)

hog_dict = {}

for pic in os.listdir(pic_path):
	if pic.split('.')[1] == 'jpg':
		pic_name = pic.split('.')[0]
		img_read = cv2.imread(os.path.join(pic_path,pic))
		img_read = cv2.resize(img_read, (128,128)) # to get the same number of features
		hog_dict[pic_name] = hog.compute(img_read, padding)
		print '√ Finish ' + pic
		
my_dictionary = {k: v.tolist() for k, v in hog_dict.items()}
hog_feature = {k: [i[0] for i in v] for k, v in my_dictionary.items()}
hog_feature = pd.DataFrame.from_dict(hog_feature, orient='index')
hog_feature.to_csv('/Users/pengfeiwang/Desktop/dogkfc/Project3_poodleKFC_train/hog_feature.csv')
# should be 2000*142884

# use pca to reduce dimention


