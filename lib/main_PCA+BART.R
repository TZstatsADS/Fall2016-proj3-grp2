

################################
####### Bart Machine ###########
################################

# PCA reduce dimensions

feature <- read.csv("/Users/zc2320/Downloads/Fall2016-proj3-grp2-master/Project3_poodleKFC_train/sift_features.csv")
X <- t(feature)
y <- rep(c(0, 1), each = 1000)
index <- c(1:2000)
testindex <- sample(index, trunc(2000/3))

##pca
df <- prcomp(X)
### pca plot
pr_var <- df$sdev ^ 2
pve <- pr_var / sum(pr_var)
plot(cumsum(pve), xlab = "Principal Component", ylab = "Cumulative Proportion of Variance Explained", ylim = c(0, 1), type = "b")

df <- as.data.frame(df$x)

data_train <- df[-testindex, ]
lab_train <- y[-testindex]

data_test <- df[testindex, ]
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
  fit_bm <- bartMachine(X = dat_train, y = label_train, num_trees = ntree, mem_cache_for_speed = FALSE)
  
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
  
  return(as.numeric(pred > 0.5))
}


# train the model with the entire training set
tm_train_bm <- system.time(bm_train <- train_bm(data_train, lab_train, 300))
save(bm_train, file="./output/bm_train.RData")
tm_train_bm[1]

### Make prediction 
tm_test_bm <- system.time(bm_test <- test_bm(bm_train, data_test))
save(bm_test, file="./output/rf_test.RData")
tm_test_bm[1]
1-sum(bm_test!=lab_test)/666
