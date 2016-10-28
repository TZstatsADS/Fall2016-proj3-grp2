import lasagne,theano,scipy
import sys,pickle,os,re
import theano.tensor as T
import numpy as np
from lasagne.nonlinearities import softmax,very_leaky_rectify
from lasagne.layers import InputLayer, DenseLayer, get_output,MaxPool2DLayer,Conv2DLayer,Layer
from lasagne.updates import sgd, apply_momentum,apply_nesterov_momentum,adagrad
from scipy.misc import *
from collections import OrderedDict
from lasagne.init import Constant, GlorotUniform

###save the data from dir to pickle file
path="/Users/pengfeiwang/Desktop/dogkfc/Project3_poodleKFC_train/images/"
imagename=filter(lambda x:re.search(r".jpg",x),os.listdir(path))
np.random.shuffle(imagename)
trainy=np.zeros(1750);testy=np.zeros(250)
trainx=np.zeros([1750,3,128,128])
testx=np.zeros([250,3,128,128])
for i1,i in enumerate(imagename[:1750]):
	if re.search("chicken",i):
		trainy[i1]=1
	trainx[i1]=imresize(imread(path+i),[128,128,3]).transpose([2,0,1])


for i1,i in enumerate(imagename[1750:]):
	if re.search("chicken",i):
		testy[i1]=1
	testx[i1]=imresize(imread(path+i),[128,128,3]).transpose([2,0,1])

pickle.dump(trainx,open("/Users/pengfeiwang/Desktop/dogkfc/Project3_poodleKFC_train/data/trainx.pkl","wb"))
pickle.dump(testx,open("/Users/pengfeiwang/Desktop/dogkfc/Project3_poodleKFC_train/data/testx.pkl","wb"))
pickle.dump(trainy,open("/Users/pengfeiwang/Desktop/dogkfc/Project3_poodleKFC_train/data/trainy.pkl","wb"))
pickle.dump(testy,open("/Users/pengfeiwang/Desktop/dogkfc/Project3_poodleKFC_train/data/testy.pkl","wb"))


###load data from file
train_x=pickle.load(open("/Users/pengfeiwang/Desktop/dogkfc/Project3_poodleKFC_train/data/trainx.pkl","rb"))
test_x=pickle.load(open("/Users/pengfeiwang/Desktop/dogkfc/Project3_poodleKFC_train/data/testx.pkl","rb"))
train_y=pickle.load(open("/Users/pengfeiwang/Desktop/dogkfc/Project3_poodleKFC_train/data/trainy.pkl","rb"))
test_y=pickle.load(open("/Users/pengfeiwang/Desktop/dogkfc/Project3_poodleKFC_train/data/testy.pkl","rb"))
train_x=train_x/256.
test_x=test_x/256.
train_y=train_y.astype("int64")
test_y=test_y.astype("int64")

rng=np.random


###start to build the CNN network
x1=T.tensor4('x1',dtype='float64')
y1=T.vector('y1',dtype='int64')
batchsize=100
l0=InputLayer(shape=(None,3,128,128),input_var=x1)
l1=Conv2DLayer(l0,48,(5,5),nonlinearity=very_leaky_rectify,W=GlorotUniform('relu'))
l2=MaxPool2DLayer(l1,(2,2))
l3=Conv2DLayer(l2,64,(5,5),nonlinearity=very_leaky_rectify,W=GlorotUniform('relu'))
l4=MaxPool2DLayer(l3,(2,2))
l5=Conv2DLayer(l4,96,(5,5),nonlinearity=very_leaky_rectify,W=GlorotUniform('relu'))
l6=MaxPool2DLayer(l5,(3,3))
l7=DenseLayer(l6,512,nonlinearity=very_leaky_rectify,W=lasagne.init.GlorotNormal())
#l7_5=cyclicpool(l7)
#l7_5=lasagne.layers.DropoutLayer(l7)
l8=DenseLayer(l7,2,nonlinearity=softmax)



rate=theano.shared(.0002)
params = lasagne.layers.get_all_params(l8)
prediction = lasagne.layers.get_output(l8)
loss = lasagne.objectives.categorical_crossentropy(prediction,y1)
loss = loss.mean()
updates_sgd = adagrad(loss, params, learning_rate=rate)
updates = apply_nesterov_momentum(updates_sgd, params, momentum=0.9)


train_model = theano.function([x1,y1],outputs=loss,updates=updates)

pred=theano.function([x1,y1],outputs=lasagne.objectives.categorical_crossentropy(prediction,y1))

###begin to train
renewtrain=len(train_x)/batchsize
renewtest=len(test_x)/batchsize
for i in range(15000):
    if i>325 and i<3000:
        rate.set_value(.001)
    elif i>6500 and i<15000:
        rate.set_value(.0005)            
    i1=i%renewtrain
    tindex=range(i1*batchsize,(i1+1)*batchsize)
    newloss=train_model(train_x[tindex],train_y[tindex])
    print 'in %d round, the loss function is %f'%(i+1,newloss) 
    if i%renewtrain==0:
        tt1=range(250)
        pred1=0.
        tmp_x=test_x[tt1]
        tmp_y=test_y[tt1]
        pred1+=sum(pred(tmp_x,tmp_y))
        print 'in %d circle, the total test error is %f'%(i/renewtest+1,pred1/250.0)
        tmp=rng.permutation(1750)
        train_x=train_x[tmp]
        train_y=train_y[tmp]
    

