# Project: Labradoodle or Fried Chicken? 
![image](https://s-media-cache-ak0.pinimg.com/236x/6b/01/3c/6b013cd759c69d17ffd1b67b3c1fbbbf.jpg)
### [Full Project Description](doc/project3_desc.html)

Term: Fall 2016

+ Team # I have a dog, I have a chicken...Ehhh...
+ Team members:  
 + Zheyuan Chen
 + Minghao Dai  
 + Rong Li 
 + Pengfei Wang 
 + Kaisheng Wang
 
+ Project summary: 

 In this project, we created a classification engine for images of poodles versus images of fried chickens.Here is the game guidance. First of all, we need to transform the visual information to some numbers which can be understood and manipulated by computers. Then, filtering and selecting features from the first stage will help us improve the computing speed and prediction rate. At the final found, we feed our selected features to a classifier to carry our images to the right destination.

 + Feature Extraction and Selection:
     
     HOG is better at capturing the global features, and identifying the shape. A large portion of pictures we have are imcomplete which makes it hard to tell from the shape.SIFT works really well with Bags of words model. Bags of words identified the objects from little key features from each category. But the only problem was, extracting raw sift features and built up bags of words model took way more than 30 mins. So we generated a codebook from the training dataset we have, so now we only need to compare the new sift features with the code book.

 + Model Selection:
     
     As for the models, we briefly tested all possibly good models with different feature extraction and selection combination. From all of these primary models, we found sift with bag of words features generally perform the best in every model. As for model selection, Bayesian additive regression tree and support vector machine have the best prediction rate.At the final round of model competition, support vector machine beat bartmachine by a slightly higher prediction rate, faster speed and less computer memory requirement.

 



|               | Hog_pca       | Ori_Sift      | Sift_bow     | Resized_Img  |
| ------------- |:-------------:|:-------------:|:------------:|:------------:|
| GBM           |     NA        |  70.65%       |   85.70%     |	NA    |
| RF            |     51.25% 	| 71.40%       	|82.60%        |        NA    |
| SVM           |  55.00%       | 74.85%        |  87.11%      |	NA    |
| XGB           |   47.85%      | 62.90%     	| 73.89%       |        NA    |
| CNN           |       NA      |    NA     	|      NA      |      92.41%  |  


+ Windows and Mac i5 16g RAM
+ R version: 3.2.3; Python version: 2.7
+ R Packages: gbm, e1071
+ Python Package: Opencv3, simplecv, Theano, lasagne

	
**Contribution statement**: 
+ Minghao Dai:
   + Tuned and run the GBM and SVM models
   + Tried to use PCA to do feature selection
   + Completed the final train.R and test.R functions


Following [suggestions](http://nicercode.github.io/blog/2013-04-05-projects/) by [RICH FITZJOHN](http://nicercode.github.io/about/#Team) (@richfitz). This folder is orgarnized as follows.

```
proj/
├── lib/
├── data/
├── doc/
├── figs/
└── output/
```

Please see each subfolder for a README file.
