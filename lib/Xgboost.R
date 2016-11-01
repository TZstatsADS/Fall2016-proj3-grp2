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
#select the number of trees
err <- vector()

for (i in 1:50){
  bst <- xgboost(data = dtrain, max.depth = 2, eta = 1, nthread = 2, nround = i, objective = "binary:logistic", verbose = 0)
  pred <- predict(bst, data_test)
  prediction <- as.numeric(pred > 0.5)
  err[i] <- mean(as.numeric(pred > 0.5) != lab_test)
}
err
plot(err, xlab = "nround", ylab = "error rate", type = "l")
Nround <- as.numeric(which.min(err))
#5
err[which.min(err)]


#select the max_depth
err <- vector()

for (i in 2:10){
  bst <- xgboost(data = dtrain, max.depth = i, eta = 1, nthread = 5, nround = Nround, objective = "binary:logistic", verbose = 0)
  pred <- predict(bst, data_test)
  prediction <- as.numeric(pred > 0.5)
  err[i] <- mean(as.numeric(pred > 0.5) != lab_test)
}
err
plot(err, xlab = "nround", ylab = "error rate", type = "l")
Max.depth <- as.numeric(which.min(err))
Max.depth
#2
err[which.min(err)]

cv.function <- function(X.train, y.train, d, K){
  
  n <- length(y.train)
  n.fold <- floor(n/K)
  s <- sample(rep(1:K, c(rep(n.fold, K-1), n-(K-1)*n.fold)))  
  cv.error <- rep(NA, K)
  
  for (i in 1:K){
    train.data <- X.train[s != i,]
    train.label <- y.train[s != i]
    test.data <- X.train[s == i,]
    test.label <- y.train[s == i]
    
    par <- list(depth=d)
    fit <- train(train.data, train.label, par)
    pred <- test(fit, test.data)  
    cv.error[i] <- mean(pred != test.label)  
    
  }			
  return(c(mean(cv.error),sd(cv.error)))
  
}