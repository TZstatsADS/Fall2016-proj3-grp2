# -*- coding: <utf-8> -*-

import cv2, os, sys
import pickle
from cv2 import *
from numpy import *
import pandas as pd


def preProcessImages(image_paths):
    descriptors= []
    for image_path in os.listdir(image_paths):
    	if image_path.split('.')[1] == 'jpg':
			im = imread(os.path.join(image_paths,image_path))
			kpts = feature_det.detect(im)
 			kpts, des = feature_det.compute(im, kpts)
			descriptors.append(des)
			print '---Finished ' + image_path + ' ---'
    return descriptors

def getImagedata(feature_det, bow_extract, path):
    im = imread(path)
    featureset = bow_extract.compute(im, feature_det.detect(im))
    return featureset


# the path to test images
image_paths = '/Project3_poodleKFC_test/images_test'  

feature_det = xfeatures2d.SIFT_create()
voc = pickle.load(open('Fall2016-proj3-grp2/data/feature_bow_voc.p','rb'))
matcher = BFMatcher(NORM_L2)
bow_extract = BOWImgDescriptorExtractor(feature_det,matcher)
bow_extract.setVocabulary(voc)

traindata = []  
for image in os.listdir(image_paths):
	if image.split('.')[1] != 'DS_Store':
	    imagepath = os.path.join(image_paths,image)
	    print imagepath
	    featureset = getImagedata(feature_det,bow_extract,imagepath)
	    traindata.append(featureset)
    
# traidata ==> 2000 * 500


# add ==> classes
trainning_set = pd.DataFrame([i.tolist() for i in traindata])
trainning_set = trainning_set[0].apply(pd.Series)
index_name = [i for i in os.listdir(image_paths) if i != '.DS_Store']
trainning_set.index = index_name
trainning_set.to_csv('/Users/pengfeiwang/Desktop/bow_test.csv')


