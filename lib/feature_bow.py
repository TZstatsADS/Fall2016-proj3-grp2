'''
    Author: http://awesomealgorithm.blogspot.com/2015/08/machine-learning-image-detection-cats.html
    Python Version: 2.7
'''
import cv2


feature_det = cv2.xfeatures2d.SIFT_create()

def getImagedata(feature_det,bow_extract,path):
    im = imread(path)
    featureset = bow_extract.compute(im, feature_det.detect(im))
    return featureset


flann_params = dict(algorithm = 1, trees = 5)     
matcher = FlannBasedMatcher(flann_params, {})
bow_extract  =BOWImgDescriptorExtractor(feature_det,matcher)
bow_train = BOWKMeansTrainer(500)
for des in descriptors:
    bow_train.add(des)

start = time.time()
voc = bow_train.cluster()
bow_extract.setVocabulary( voc )
end = time.time()
print("minutes spent in creating Vocabulary")
print((end - start)/60)


traindata = []  
start = time.time()
for imagepath in image_paths:
    featureset = getImagedata(feature_det,bow_extract,imagepath)
    traindata.extend(featureset)
