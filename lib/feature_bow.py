'''
    Author: http://awesomealgorithm.blogspot.com/2015/08/machine-learning-image-detection-cats.html
    Python Version: 2.7
'''

import cv2, os
from cv2 import *


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

feature_det = cv2.xfeatures2d.SIFT_create()
image_paths = ''


bow_train = BOWKMeansTrainer(500)  # clusterCount = 500

descriptors = preProcessImages(image_paths)
for des in descriptors:
    bow_train.add(des)

voc = bow_train.cluster()


flann_params = dict(algorithm = 1, trees = 5)     
matcher = FlannBasedMatcher(flann_params, {})
bow_extract = BOWImgDescriptorExtractor(feature_det,matcher)
bow_extract.setVocabulary(voc)


traindata = []  
for imagepath in image_paths:
    featureset = getImagedata(feature_det,bow_extract,imagepath)
    traindata.append(featureset)
    
