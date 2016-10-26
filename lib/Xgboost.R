install.packages("xgboost")
require(xgboost)
require(methods)
require(data.table)
require(magrittr)
library(xgboost)
library(methods)
library(data.table)
library(magrittr)
load("/Users/kaishengwang/Downloads/traintest_data.RData")
dim(data_train)
dim(data_test)


numberOfClasses <- 2

param <- list("objective" = "multi:softprob",
              "eval_metric" = "mlogloss",
              "num_class" = numberOfClasses)

cv.nround <- 5
cv.nfold <- 3
bst.cv = xgb.cv(param=param, data = data_train, label = lab_train, 
                nfold = cv.nfold, nrounds = cv.nround)

nround = 50
bst = xgboost(param=param, data = data_train, label = lab_train, nrounds=nround)
