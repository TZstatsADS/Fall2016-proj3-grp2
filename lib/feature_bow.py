# -*- coding: <utf-8> -*-

import cv2, os
from cv2 import *
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
image_paths = '/Users/pengfeiwang/Desktop/dogkfc/Project3_poodleKFC_train/images/'


train_set = pd.read_csv('')  # training set dir
classes_train_set = train_set[['']]   # name of class


bow_train = BOWKMeansTrainer(500)  # set the clusterCount => 500
descriptors = preProcessImages(image_paths)

# delete the none value
descriptors_not_valid = []
for i, j in enumerate(descriptors):
    if j == None:
        descriptors_not_valid.append(i)
        
descriptors_valid = [i for i in descriptors if i!= None]
classes_train_set_valid = [classes_train_set[i] for i in range(len(classes_train_set)) if i not in descriptors_not_valid]
image_valid = [os.listdir(image_paths)[i] for i in range(len(os.listdir(image_paths))) if i not in descriptors_not_valid]


for des in descriptors_valid:
    bow_train.add(des)

voc = bow_train.cluster()


matcher = BFMatcher(NORM_L2)
bow_extract = BOWImgDescriptorExtractor(feature_det,matcher)
bow_extract.setVocabulary(voc)


traindata = []  
for image in image_valid:
    imagepath = os.path.join(image_paths,image)
    featureset = getImagedata(feature_det,bow_extract,imagepath)
    traindata.append(featureset)
    
# traidata ==> 2000 * 500
# add ==> classes_train_set_valid

trainning_set = pd.DataFrame(traidata)
trainning_set['class'] = classes_train_set_valid
trainning_set.to_csv('/Users/pengfeiwang/Desktop/bow_traing.csv')


