#########################################################
### Train a classification model with training images ###
#########################################################

### Author: Group 2
### Project 3
### ADS Fall 2016

train <- function(feature.adv, feature.baseline, label_train){
  
  ### Train a Gradient Boosting Model (GBM) using processed features from training images

  ### Input: 
  ###  -  processed features in Feature_eval.RData
  ###  -  class labels for training images
  ### Output: two model object
  
  ### load libraries
  library("gbm")
  library("e1071")
  
  ### change names
  X.a <- t(feature.adv)
  X.b <- t(feature.baseline)
  y <- label_train
  
  ######################
  ### Baseline Model ###
  ### Train with gradient boosting model using feature.baseline
  
  ### Model selection with cross-validation
  # Choosing between different values of interaction depth for GBM
  source("./final version/cross_validation.R")
  source("./final version/train_gbm.R")
  source("./final version/test_gbm.R")
  
  depth_values <- seq(3, 11, 2)
  nm_values <- c(5, 10, 20)
  err.cv <- matrix(NA, length(depth_values), length(nm_values))
  K <- 5  # number of CV folds
  #we took a day to tune here
  for(i in 1:length(depth_values)){
    cat("i=", i, "\n")
    d <- depth_values[i]
    for (j in 1:length(nm_values)){
      cat("j=", j, "\n")
      nm <- nm_values[j]
      par <- list(depth=d, n.minobsinnode=nm)
      err.cv[i, j] <- cv.function(X.b, y, par, K, train_gbm, test_gbm)
    }
  }
  
  p.min <- which(err.cv == min(err.cv), arr.ind = TRUE)
  par.b.best <- list(depth = depth_values[p.min[1]], 
                     n.minobsinnode = nm_values[p.min[2]])
  
  tm_train_gbm <- system.time(gbm_train <- train_gbm(X.b, y, par.b.best))
  
  

  #####################
  ### Advance Model ###
  ### Train with support vector machine using feature.adv
  
  ### Model selection with cross-validation
  # Choosing between different values of interaction depth for GBM
  source("./final version/train_svm.R")
  source("./final version/test_svm.R")
  
  # Choosing between different values of cost C for Linear SVM 
  C <- c(1, 10, 100, 500, 1000)
  err_cv_svm_c <- rep(NA, length(C))
  K <- 5  # number of CV folds
  for(k in 1:length(C)){
    cat("k=", k, "\n")
    par <- list(kernel = "linear", cost = C[k])
    err_cv_svm_c[k] <- cv.function(X.a, y, par, K, train_svm, test_svm)
  }
  
  # Choose the best parameter value
  C_best <- C[which.min(err_cv_svm_c)]
  
  
  # Choosing between different values of gamma G for RBF SVM
  G <- c(0.01, 0.001, 0.0001, 0.00001)
  err_cv_svm_g <- rep(NA, length(G))
  K <- 5  # number of CV folds
  for(k in 1:length(G)){
    cat("k=", k, "\n")
    par <- list(kernel = "radial", cost = C_best, gamma = G[k])
    err_cv_svm_g[k] <- cv.function(X.a, y, par, K, train_svm, test_svm)
  }
  
  # Choose the best parameter value
  G_best <- G[which.min(err_cv_svm_g)]
  # Two best parameters 
  par.a.best <- list(kernel = "radial", cost = C_best, gamma = G_best)
  
  tm_train_svm <- system.time(svm_train <- train_svm(X.a, y, par.a.best))
  
  
  #################################
  #return two trained model objects
  
  return(ba.train = list(baseline = gbm_train, advance = svm_train))
}



