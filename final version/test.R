######################################################
### Fit the classification model with testing data ###
######################################################

### Author: Group 2
### Project 3
### ADS Fall 2016

test <- function(ba.train, feature.test){
  
  ### Fit the classfication model with testing data

  ### Input: 
  ###  -  the two trained objects using training data
  ###  -  processed features from testing images 
  ### Output: prediction labels
  
  ### load libraries
  library("gbm")
  library("e1071")
  
  ### change names
  X.t <- t(feature.test)
  
  ######################
  ### Baseline Model ###
  source("./final version/test_gbm.R")
  
  gbm_test <- test_gbm(ba.train$baseline, X.t)
  
  #####################
  ### Advance Model ###
  source("./final version/test_svm.R")
  
  svm_test <- test_svm(ba.train$advance, X.t)
  
  #################################
  #return two trained model objects
  
  return(ba.test = list(baseline = gbm_test, advance = svm_test))
  
}

