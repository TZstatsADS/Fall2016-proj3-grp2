
feature <- read.csv("sift_features.csv")
X <- t(feature)
y <- rep(c(0, 1), each = 1000)
index <- c(1:2000)
testindex <- sample(index, trunc(2000/3))

data_train <- X[-testindex, ]
lab_train <- y[-testindex]

data_test <- X[testindex, ]
lab_test <- y[testindex]


#######################
####### GBM ###########
#######################


### Train a classification model with training images
source("./lib/train.R")
source("./lib/test.R")


### Model selection with cross-validation
# Choosing between different values of interaction depth for GBM
source("./lib/cross_validation.R")
depth_values <- seq(3, 11, 2)
err_cv <- array(dim=c(length(depth_values), 2))
K <- 5  # number of CV folds
for(k in 1:length(depth_values)){
  cat("k=", k, "\n")
  err_cv[k,] <- cv.function(data_train, lab_train, depth_values[k], K, 
                            train_gbm, test_gbm)
}
save(err_cv, file="./output/gbm_err_cv.RData")

# Visualize CV results
pdf("./figs/cv_results_gbm.pdf", width=7, height=5)
plot(depth_values, err_cv[,1], xlab="Interaction Depth", ylab="CV Error",
     main="GBM Cross Validation Error", type="n", ylim=c(0, 0.5))
points(depth_values, err_cv[,1], col="blue", pch=16)
lines(depth_values, err_cv[,1], col="blue")
arrows(depth_values, err_cv[,1]-err_cv[,2],depth_values, err_cv[,1]+err_cv[,2], 
       length=0.1, angle=90, code=3)
dev.off()

# Choose the best parameter value
depth_best <- depth_values[which.min(err_cv[,1])]
par_best <- list(depth=depth_best)


# train the model with the entire training set
tm_train_gbm <- system.time(gbm_train <- train_gbm(data_train, lab_train, par_best))
save(gbm_train, file="./output/gbm_train.RData")


### Make prediction 
tm_test_gbm <- system.time(gbm_test <- test_gbm(gbm_train, data_test))
save(gbm_test, file="./output/gbm_test.RData")
1-sum(gbm_test!=lab_test)/666



################################
####### Random Forest###########
################################



train_rf <- function(dat_train, label_train, ntry, ntree){
  
  ### Input: 
  ###  -  processed features from images 
  ###  -  numbers of size of sampled varibles for each individual tree
  ###  -  numbers of individual trees
  ### Output: training model specification
  
  ### load libraries
  library("randomForest")
  library("dplyr")
  set.seed(123)
  ### Train with SVM model
  fit_rf <- randomForest(label_train ~ . , data = dat_train, mtry= ntry, ntree = ntree, importance = TRUE)
  
  return(fit_rf)
}

test_rf <- function(fit_train, dat_test){
  
  ### Fit the classfication model with testing data
  
  ### Input: 
  ###  - the fitted classification model using training data
  ###  -  processed features from testing images 
  ### Output: training model specification
  
  library("randomForest")
  pred <- predict(fit_train, newdata = dat_test)
  
  return(as.numeric(pred > 0.5))
}


# train the model with the entire training set
tm_train_rf <- system.time(rf_train <- train_rf(data_train, lab_train, 200, 2000))
save(rf_train, file="./output/rf_train.RData")
tm_train_rf[1]

### Make prediction 
tm_test_rf <- system.time(rf_test <- test_rf(rf_train, data_test))
save(rf_test, file="./output/rf_test.RData")
tm_test_rf[1]
1-sum(rf_test!=lab_test)/666

