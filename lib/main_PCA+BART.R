

################################
####### Bart Machine ###########
################################

# PCA reduce dimensions

feature <- read.csv("./data/feature_sift_bow.csv")
X <- feature[2:501]
y <- rep(c(1, 0), each = 1000)
index <- c(1:2000)
set.seed(124)
testindex <- sample(index, 400)

data_train <- X[-testindex, ]
lab_train <- y[-testindex]

data_test <- X[testindex, ]
lab_test <- y[testindex]

### Bart Machine

train_bm <- function(dat_train, label_train, ntree){
  
  ### Input: 
  ###  -  processed features from images 
  ### Output: training model specification
  
  ### load libraries
  library("bartMachine")
  library("dplyr")
  
  set.seed(123)
  set_bart_machine_num_cores(parallel::detectCores())
  gc();Sys.sleep(10);gc()
  fit_bm <- bartMachine(X = dat_train, y = label_train, num_trees = ntree, 
                          mem_cache_for_speed = FALSE)
  
  return(fit_bm)
}

test_bm <- function(fit_train, dat_test){
  
  ### Fit the classfication model with testing data
  
  ### Input: 
  ###  - the fitted classification model using training data
  ###  -  processed features from testing images 
  ### Output: training model specification
  
  library("bartMachine")
  
  pred <- predict(fit_train, new_data = dat_test)
  
  return(as.numeric(pred > mean(pred)))
}

# train the model with the entire training set
tm_train_bm <- system.time(bm_train <- train_bm(X, y, 200))
save(bm_train, file="./output/bm_train.RData")
tm_train_bm[1]

### Make prediction 
tm_test_bm <- system.time(bm_test <- test_bm(bm_train, data_test))
save(bm_test, file="./output/bm_test.RData")
tm_test_bm[1]
1-sum(bm_test!=lab_test)/400
rm()