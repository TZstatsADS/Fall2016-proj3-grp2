
install.packages("xgboost")
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
set.seed(1)
load("/Users/kaishengwang/Downloads/traintest_data.RData")
dim(data_train)
dim(data_test)
lab_test[lab_test == 1] <- 2
lab_test[lab_test == 0] <- 1
lab_test[lab_test == 2] <- 0

lab_train[lab_train == 1] <- 2
lab_train[lab_train == 0] <- 1
lab_train[lab_train == 2] <- 0



numberOfClasses <- 2

param <- list("objective" = "binary:logistic",
              "eval_metric" = "merror",
              "num_class" = numberOfClasses)

cv.nround <- 5
cv.nfold <- 3
bst.cv = xgb.cv(param=param, data = data_train, label = lab_train, 
                nfold = cv.nfold, nrounds = cv.nround)

nround = 50
bst = xgboost(param=param, data = data_train, label = lab_train, nrounds=nround)

y_pred <- predict(bst, data_test)
y_pred

prediction <- as.numeric(y_pred > 0.5)

table(prediction, lab_test)
mean(prediction == lab_test)
