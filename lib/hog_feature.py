import cv2, os, sys
import pandas as pd
sys.path.append('/Users/pengfeiwang/Documents/anaconda2/lib/python2.7/site-packages')


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
		hog_dict[pic_name] = hog.compute(img_read, padding)
		print '√ Finish ' + pic
		break

hog_feature = pd.DataFrame.from_dict(hog_dict)
hog_feature.to_csv('/Users/pengfeiwang/Desktop/dogkfc/Project3_poodleKFC_train/hog_feature.csv')

# use pca to reduce dimention


