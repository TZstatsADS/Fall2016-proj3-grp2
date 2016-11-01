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

y <- t(lab_train)
y

numberOfClasses <- 2

param <- list("objective" = "binary:logistic",
              "eval_metric" = "merror",
              "num_class" = numberOfClasses)

cv.nround <- 10
cv.nfold <- 5

bst.cv = xgb.cv(param=param, data = data_train, label = y, 
                nfold = cv.nfold, nrounds = cv.nround)

nround <- 50
bst = xgboost(param=param, data = data_train, label = y, nrounds=nround)
