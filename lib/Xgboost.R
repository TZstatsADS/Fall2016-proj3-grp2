install.packages("drat", repos="https://cran.rstudio.com")
drat:::addRepo("dmlc")
install.packages("xgboost", repos="http://dmlc.ml/drat/", type = "source")
require(xgboost)
require(methods)
require(data.table)
require(magrittr)
library(xgboost)
library(methods)
library(data.table)
library(magrittr)
library(readr)
library(stringr)
library(caret)
library(car)
data_whole <- fread("/Users/kaishengwang/Desktop/Applied\ Data\ Science\ Project\ 3/sift_features.csv", header = T, stringsAsFactors = F)
X <- t(data_whole)
y <- rep(c(1,0), each = 1000)
index <- c(1:2000)
testindex <- sample(index, trunc(2000/5))

data_train <- X[-testindex,]
lab_train <- y[-testindex]

data_test <- X[testindex,]
lab_test <- y[testindex]

#labels <- c(matrix(data = 1, nrow = 1, ncol = 1000), matrix(data = 0, nrow = 1, ncol = 1000))
#cross validation, splid the data into 5-folders
#k <- 5
#set.seed(1)
#folds <- sample(1:k, ncol(data_whole), replace = TRUE)

#data_train <- data_whole[,folds == 1]
#data_test <- data_whole[,which(folds == 5)]

dim(data_train)
dim(data_test)
str(data_train)

bstSparse <- xgboost(data = data_train, label = lab_train, max.depth = 2, eta = 1, nthread = 2, nround = 2, objective = "binary:logistic")

bstDense <- xgboost(data = as.matrix(data_train), label = lab_train, max.depth = 2, eta = 1, nthread = 2, nrounds = 2, objective = "binary:logistic")

dtrain <- xgb.DMatrix(data = data_train, label = lab_train)
bstDMatrix <- xgboost(data = dtrain, max.depth = 2, eta = 1, nthread = 2, nround = 2, objective = "binary:logistic")

bst <- xgboost(data = dtrain, max.depth = 2, eta = 1, nthread = 2, nround = 50, objective = "binary:logistic", verbose = 0)
bst <- xgboost(data = dtrain, max.depth = 2, eta = 1, nthread = 2, nround = 2, objective = "binary:logistic", verbose = 1)
bst <- xgboost(data = dtrain, max.depth = 2, eta = 1, nthread = 2, nround = 2, objective = "binary:logistic", verbose = 2)

pred <- predict(bst, data_test)
print(length(pred))
print(head(pred))

prediction <- as.numeric(pred > 0.5)
print(head(prediction))

err <- mean(as.numeric(pred > 0.5) != lab_test)
print(paste("test-error=", err))

#select the best parameter
#select eta
set.seed(1)
error_rate <- vector()
for (i in 1:10){
  cv.res <- xgb.cv(data = data_train, label = lab_train, max.depth = 2, eta = i, min_child_weight = 1, nround = 29, subsample = 1,colsample_bytree = 1, objective = "binary:logistic", nfold = 5)
  error_rate[i] <- mean(cv.res$test.error.mean)
  }
error_rate
plot(error_rate, xlab = "eta", ylab = "error rate", type = "l")
Eta <- as.numeric(which.min(error_rate))
Eta
#1
error_rate[which.min(error_rate)]

#select the max_depth
set.seed(2)
error_rate <- vector()
for (i in 2:10){
  cv.res <- xgb.cv(data = data_train, label = lab_train, max.depth = i, eta = Eta, min_child_weight = 1, nround = 29, subsample = 1, colsample_bytree = 1, objective = "binary:logistic", nfold = 5)
  error_rate[i] <- mean(cv.res$test.error.mean)
}
error_rate
plot(error_rate, xlab = "max.depth", ylab = "error rate", type = "l")
Max.depth <- as.numeric(which.min(error_rate))
Max.depth
#7
error_rate[which.min(error_rate)]

#select the Min_child_weight
set.seed(3)
error_rate <- vector()
for (i in 1:20){
  cv.res <- xgb.cv(data = data_train, label = lab_train, max.depth = Max.depth, eta = Eta, min_child_weight = i, nround = 29,subsample = 1, colsample_bytree = 1, objective = "binary:logistic", nfold = 5)
  error_rate[i] <- mean(cv.res$test.error.mean)
}
error_rate
plot(error_rate, xlab = "min_child_weight", ylab = "error rate", type = "l")
Min_child_weight <- as.numeric(which.min(error_rate))
Min_child_weight
#8
error_rate[which.min(error_rate)]

#select the Subsample
set.seed(4)
error_rate <- vector()
for (i in 1:30){
  cv.res <- xgb.cv(data = data_train, label = lab_train, max.depth = Max.depth, eta = Eta, min_child_weight = Min_child_weight, nround = 29,subsample = i, colsample_bytree = 1, objective = "binary:logistic", nfold = 5)
  error_rate[i] <- mean(cv.res$test.error.mean)
}
error_rate
plot(error_rate, xlab = "subsample", ylab = "error rate", type = "l")
Subsample <- as.numeric(which.min(error_rate))
Subsample
#15
error_rate[which.min(error_rate)]

#select the colsample_bytree
set.seed(5)
error_rate <- vector()
for (i in 1:10){
  cv.res <- xgb.cv(data = data_train, label = lab_train, max.depth = Max.depth, eta = Eta, min_child_weight = Min_child_weight, nround = 29,subsample = Subsample, colsample_bytree = i, objective = "binary:logistic", nfold = 5)
  error_rate[i] <- mean(cv.res$test.error.mean)
}
error_rate
plot(error_rate, xlab = "colsample_bytree", ylab = "error rate", type = "l")
Colsample_bytree <- as.numeric(which.min(error_rate))
Colsample_bytree
#9
error_rate[which.min(error_rate)]

#The selected parmeters are:
#eta = 1
#Max.depth = 7
#Min_child_weight = 8
#Subsample = 15
#Colsample_bytree = 9

cv.res <- xgb.cv(data = data_test, label = lab_test, max.depth = 7, eta = 1, min_child_weight = 8, nround = 29,subsample = 15, colsample_bytree = 9, objective = "binary:logistic", nfold = 5)
error_rate <- mean(cv.res$test.error.mean)
error_rate
1-error_rate
#62.90%

#Use PCA

data_whole <- read.csv("/Users/kaishengwang/Desktop/Applied\ Data\ Science\ Project\ 3/feature_hog_pca.csv")
X <- data_whole[,2:701]
y <- rep(c(1,0), each = 1000)
index <- c(1:2000)
testindex <- sample(index, trunc(2000/5))

data_train <- X[-testindex,]
lab_train <- y[-testindex]

data_test <- X[testindex,]
lab_test <- y[testindex]
dim(data_train)
dim(data_test)

data_train <- as.matrix(data_train)
data_test <- as.matrix(data_test)

bstSparse <- xgboost(data = data_train, label = lab_train, max.depth = 2, eta = 1, nthread = 2, nround = 2, objective = "binary:logistic")

bstDense <- xgboost(data = as.matrix(data_train), label = lab_train, max.depth = 2, eta = 1, nthread = 2, nrounds = 2, objective = "binary:logistic")

dtrain <- xgb.DMatrix(data = data_train, label = lab_train)
bstDMatrix <- xgboost(data = dtrain, max.depth = 2, eta = 1, nthread = 2, nround = 2, objective = "binary:logistic")

bst <- xgboost(data = dtrain, max.depth = 2, eta = 1, nthread = 2, nround = 50, objective = "binary:logistic", verbose = 0)
bst <- xgboost(data = dtrain, max.depth = 2, eta = 1, nthread = 2, nround = 2, objective = "binary:logistic", verbose = 1)
bst <- xgboost(data = dtrain, max.depth = 2, eta = 1, nthread = 2, nround = 2, objective = "binary:logistic", verbose = 2)

pred <- predict(bst, data_test)
print(length(pred))
print(head(pred))

prediction <- as.numeric(pred > 0.5)
print(head(prediction))

err <- mean(as.numeric(pred > 0.5) != lab_test)
print(paste("test-error=", err))

#select the best parameter
#select eta
set.seed(1)
error_rate <- vector()
for (i in 1:20){
  cv.res <- xgb.cv(data = data_train, label = lab_train, max.depth = 2, eta = i, min_child_weight = 1, nround = 29, subsample = 1,colsample_bytree = 1, objective = "binary:logistic", nfold = 5)
  error_rate[i] <- mean(cv.res$test.error.mean)
}
error_rate
plot(error_rate, xlab = "eta", ylab = "error rate", type = "l")
Eta <- as.numeric(which.min(error_rate))
Eta
#15
error_rate[which.min(error_rate)]

#select the max_depth
set.seed(2)
error_rate <- vector()
for (i in 2:10){
  cv.res <- xgb.cv(data = data_train, label = lab_train, max.depth = i, eta = Eta, min_child_weight = 1, nround = 29, subsample = 1, colsample_bytree = 1, objective = "binary:logistic", nfold = 5)
  error_rate[i] <- mean(cv.res$test.error.mean)
}
error_rate
plot(error_rate, xlab = "max.depth", ylab = "error rate", type = "l")
Max.depth <- as.numeric(which.min(error_rate))
Max.depth
#8
error_rate[which.min(error_rate)]

#select the Min_child_weight
set.seed(3)
error_rate <- vector()
for (i in 1:20){
  cv.res <- xgb.cv(data = data_train, label = lab_train, max.depth = Max.depth, eta = Eta, min_child_weight = i, nround = 29,subsample = 1, colsample_bytree = 1, objective = "binary:logistic", nfold = 5)
  error_rate[i] <- mean(cv.res$test.error.mean)
}
error_rate
plot(error_rate, xlab = "min_child_weight", ylab = "error rate", type = "l")
Min_child_weight <- as.numeric(which.min(error_rate))
Min_child_weight
#12
error_rate[which.min(error_rate)]

#select the Subsample
set.seed(4)
error_rate <- vector()
for (i in 1:30){
  cv.res <- xgb.cv(data = data_train, label = lab_train, max.depth = Max.depth, eta = Eta, min_child_weight = Min_child_weight, nround = 29,subsample = i, colsample_bytree = 1, objective = "binary:logistic", nfold = 5)
  error_rate[i] <- mean(cv.res$test.error.mean)
}
error_rate
plot(error_rate, xlab = "subsample", ylab = "error rate", type = "l")
Subsample <- as.numeric(which.min(error_rate))
Subsample
#19
error_rate[which.min(error_rate)]

#select the colsample_bytree
set.seed(5)
error_rate <- vector()
for (i in 1:10){
  cv.res <- xgb.cv(data = data_train, label = lab_train, max.depth = Max.depth, eta = Eta, min_child_weight = Min_child_weight, nround = 29,subsample = Subsample, colsample_bytree = i, objective = "binary:logistic", nfold = 5)
  error_rate[i] <- mean(cv.res$test.error.mean)
}
error_rate
plot(error_rate, xlab = "colsample_bytree", ylab = "error rate", type = "l")
Colsample_bytree <- as.numeric(which.min(error_rate))
Colsample_bytree
#9
error_rate[which.min(error_rate)]

#The selected parmeters are:
#eta = 15
#Max.depth = 8
#Min_child_weight = 12
#Subsample = 19
#Colsample_bytree = 9

cv.res <- xgb.cv(data = data_test, label = lab_test, max.depth = 8, eta = 15, min_child_weight = 12, nround = 29,subsample = 19, colsample_bytree = 9, objective = "binary:logistic", nfold = 5)
error_rate <- mean(cv.res$test.error.mean)
error_rate
1-error_rate
#45.75%

#BOW

data_whole <- read.csv("/Users/kaishengwang/Desktop/Applied\ Data\ Science\ Project\ 3/feature_sift_bow.csv")
X <- data_whole[,2:501]
y <- rep(c(1,0), each = 1000)
index <- c(1:2000)
testindex <- sample(index, trunc(2000/5))

data_train <- X[-testindex,]
lab_train <- y[-testindex]

data_test <- X[testindex,]
lab_test <- y[testindex]
dim(data_train)
dim(data_test)

data_train <- as.matrix(data_train)
data_test <- as.matrix(data_test)

bstSparse <- xgboost(data = data_train, label = lab_train, max.depth = 2, eta = 1, nthread = 2, nround = 2, objective = "binary:logistic")

bstDense <- xgboost(data = as.matrix(data_train), label = lab_train, max.depth = 2, eta = 1, nthread = 2, nrounds = 2, objective = "binary:logistic")

dtrain <- xgb.DMatrix(data = data_train, label = lab_train)
bstDMatrix <- xgboost(data = dtrain, max.depth = 2, eta = 1, nthread = 2, nround = 2, objective = "binary:logistic")

bst <- xgboost(data = dtrain, max.depth = 2, eta = 1, nthread = 2, nround = 50, objective = "binary:logistic", verbose = 0)
bst <- xgboost(data = dtrain, max.depth = 2, eta = 1, nthread = 2, nround = 2, objective = "binary:logistic", verbose = 1)
bst <- xgboost(data = dtrain, max.depth = 2, eta = 1, nthread = 2, nround = 2, objective = "binary:logistic", verbose = 2)

pred <- predict(bst, data_test)
print(length(pred))
print(head(pred))

prediction <- as.numeric(pred > 0.5)
print(head(prediction))

err <- mean(as.numeric(pred > 0.5) != lab_test)
print(paste("test-error=", err))

#select the best parameter
#select eta
set.seed(1)
error_rate <- vector()
for (i in 1:20){
  cv.res <- xgb.cv(data = data_train, label = lab_train, max.depth = 2, eta = i, min_child_weight = 1, nround = 29, subsample = 1,colsample_bytree = 1, objective = "binary:logistic", nfold = 5)
  error_rate[i] <- mean(cv.res$test.error.mean)
}
error_rate
plot(error_rate, xlab = "eta", ylab = "error rate", type = "l")
Eta <- as.numeric(which.min(error_rate))
Eta
#1
error_rate[which.min(error_rate)]

#select the max_depth
set.seed(2)
error_rate <- vector()
for (i in 2:10){
  cv.res <- xgb.cv(data = data_train, label = lab_train, max.depth = i, eta = Eta, min_child_weight = 1, nround = 29, subsample = 1, colsample_bytree = 1, objective = "binary:logistic", nfold = 5)
  error_rate[i] <- mean(cv.res$test.error.mean)
}
error_rate
plot(error_rate, xlab = "max.depth", ylab = "error rate", type = "l")
Max.depth <- as.numeric(which.min(error_rate))
Max.depth
#4
error_rate[which.min(error_rate)]

#select the Min_child_weight
set.seed(3)
error_rate <- vector()
for (i in 1:20){
  cv.res <- xgb.cv(data = data_train, label = lab_train, max.depth = Max.depth, eta = Eta, min_child_weight = i, nround = 29,subsample = 1, colsample_bytree = 1, objective = "binary:logistic", nfold = 5)
  error_rate[i] <- mean(cv.res$test.error.mean)
}
error_rate
plot(error_rate, xlab = "min_child_weight", ylab = "error rate", type = "l")
Min_child_weight <- as.numeric(which.min(error_rate))
Min_child_weight
#6
error_rate[which.min(error_rate)]

#select the Subsample
set.seed(4)
error_rate <- vector()
for (i in 1:30){
  cv.res <- xgb.cv(data = data_train, label = lab_train, max.depth = Max.depth, eta = Eta, min_child_weight = Min_child_weight, nround = 29,subsample = i, colsample_bytree = 1, objective = "binary:logistic", nfold = 5)
  error_rate[i] <- mean(cv.res$test.error.mean)
}
error_rate
plot(error_rate, xlab = "subsample", ylab = "error rate", type = "l")
Subsample <- as.numeric(which.min(error_rate))
Subsample
#15
error_rate[which.min(error_rate)]

#select the colsample_bytree
set.seed(5)
error_rate <- vector()
for (i in 1:10){
  cv.res <- xgb.cv(data = data_train, label = lab_train, max.depth = Max.depth, eta = Eta, min_child_weight = Min_child_weight, nround = 29,subsample = Subsample, colsample_bytree = i, objective = "binary:logistic", nfold = 5)
  error_rate[i] <- mean(cv.res$test.error.mean)
}
error_rate
plot(error_rate, xlab = "colsample_bytree", ylab = "error rate", type = "l")
Colsample_bytree <- as.numeric(which.min(error_rate))
Colsample_bytree
#2
error_rate[which.min(error_rate)]

#The selected parmeters are:
#eta = 1
#Max.depth = 4
#Min_child_weight = 6
#Subsample = 15
#Colsample_bytree = 2

cv.res <- xgb.cv(data = data_test, label = lab_test, max.depth = 4, eta = 1, min_child_weight = 6, nround = 29,subsample = 15, colsample_bytree = 2, objective = "binary:logistic", nfold = 5)
error_rate <- mean(cv.res$test.error.mean)
error_rate
1-error_rate
#73.89%
