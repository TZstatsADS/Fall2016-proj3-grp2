# -*- coding: <utf-8> -*-

import cv2, os, sys
from cv2 import *
from numpy import *
sys.path.append('/Users/pw2406/anaconda/lib/python2.7/site-packages')
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

feature_det = xfeatures2d.SIFT_create()
image_paths = '/Users/pw2406/Desktop/Project3_poodleKFC_train/images/'


bow_train = BOWKMeansTrainer(500)  # set the clusterCount => 500
descriptors = preProcessImages(image_paths)


for des in descriptors:
    bow_train.add(des)

voc = bow_train.cluster()
matcher = BFMatcher(NORM_L2)
bow_extract = BOWImgDescriptorExtractor(feature_det,matcher)
bow_extract.setVocabulary(voc)


traindata = []  
for image in os.listdir(image_paths):
    imagepath = os.path.join(image_paths,image)
    print imagepath
    featureset = getImagedata(feature_det,bow_extract,imagepath)
    traindata.append(featureset)
    
# traidata ==> 2000 * 500
# add ==> classes

trainning_set = pd.DataFrame([i.tolist() for i in traindata])
trainning_set = trainning_set[0].apply(pd.Series)
trainning_set.to_csv('/Users/pw2406/Desktop/bow_traing.csv')


