# -*- coding: utf-8 -*-
import cv2, os, sys
sys.path.append('/Users/pengfeiwang/Documents/anaconda2/lib/python2.7/site-packages')
import pandas as pd
import numpy as np

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
## should be 2000*142884, around 4.3G


# use pca to reduce dimention
from sklearn.decomposition import PCA
dta = pd.read_csv('/Users/pengfeiwang/Desktop/dogkfc/Project3_poodleKFC_train/hog_feature.csv')
dta = dta.drop(dta.columns[[0]], axis=1) 

## when n_components=500, the sum of information included is about 85%
## when n_components=700, the sum of information included is about 90%
## ==> choose n_components=700

pca = PCA(n_components=700)   
pca.fit(dta)
print(pca.explained_variance_ratio_)
sum(pca.explained_variance_ratio_[:700])

dta = pca.transform(test_set)
dta = np.matrix(dta)
dta_new = pd.DataFrame(dta)
index_name = os.listdir(pic_path)
dta_new.index = index_name
dta_new.to_csv('/Users/pengfeiwang/Desktop/hog_pca.csv')


# Agglomerative Clustering
# from sklearn.cluster import AgglomerativeClustering
# from sklearn.metrics import adjusted_rand_score
# for i in range(5):
#     model = AgglomerativeClustering(n_clusters=n_clusters)
#     adjusted_rand_score(labels_true, labels_pred)  
    
   
