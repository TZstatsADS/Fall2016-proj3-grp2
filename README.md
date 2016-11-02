# Project: Labradoodle or Fried Chicken? 
![image](https://s-media-cache-ak0.pinimg.com/236x/6b/01/3c/6b013cd759c69d17ffd1b67b3c1fbbbf.jpg)
### [Full Project Description](doc/project3_desc.html)

Term: Fall 2016

+ Team #
+ Team members
	+ Chen, Zheyuan
	+ Dai, Minghao
	+ Li, Rong rong
	+ Wang, Pengfei
	+ Wang, Kaisheng
+ Project summary: In this project, we created a classification engine for images of poodles versus images of fried chickens. 



|               | Hog_pca       | Ori_Sift      | Sift_bow     | Resized_Img  |
| ------------- |:-------------:| :------------:|:------------:|:------------:|
| GBM           |               |               |              |	NA    |
| RF            |      	        |           	|     	       |        NA    |
| SVM           |               |           	|              |	NA    |
| XGB           |               |           	| 	       |        NA    |
| CNN           |       NA      |    NA     	|      NA      |      92.41%  |  


+ Windows and Mac machine
+ R version: 3.2.3; Python version: 2.7
+ Packages: gbm, e1071

	
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
