setwd("~/GitHub/Fall2016-proj3-grp2")
feature <- read.csv("sift_features.csv")
X <- t(feature)
y <- rep(c(1, 0), each = 1000)
index <- c(1:2000)
testindex <- sample(index, trunc(2000/3))

data_train <- X[-testindex, ]
lab_train <- y[-testindex]

data_test <- X[testindex, ]
lab_test <- y[testindex]

save(data_train, lab_train, data_test, lab_test, file="./data/traintest_data.RData")


#######################
####### GBM ###########
#######################


### Train a classification model with training images
source("./lib/train.R")
source("./lib/test.R")
### feature selection
Xpca <- prcomp(X)
data_test_pca <- t(apply(data_test, 1, function(x) 
  x-as.vector(ft.pca$center))) %*% 
  as.matrix(ft.pca$rotation)

cu <- summary(Xpca)$importance[3,]
npc <- min(which(cu.var > cutoff_best))
X.select <- as.data.frame(ft.pca$x)[,1:npc]



### Model selection with cross-validation
# Choosing between different values of interaction depth for GBM
source("./lib/cross_validation.R")
depth_values <- seq(3, 11, 2)
shrinkage_values <- c(0.0001, 0.001, 0.005)
#ntree_values <- c(50, 100, 200)
nm_values <- c(5, 10, 20)
#err_cv <- array(dim=c(length(depth_values), 2))
err.cv <- matrix(NA, 3, 3)
K <- 5  # number of CV folds
for(k in 1:5){
  cat("k=", k, "\n")
  d <- depth_values[k]
  for (j in 1:3){
    cat("j=", j, "\n")
    nm <- nm_values[j]
    err.cv[k, j] <- cv.function(X, y, d, s, nm, K, train_gbm, test_gbm)
  }
}
save(err.cv, file="./output/gbm_err_cv_3.RDate")
save(err_cv, file="./output/gbm_err_cv_depth.RData")

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
tm_train_gbm <- system.time(gbm_train <- train_gbm(data_train, lab_train, par))
save(gbm_train, tm_train_gbm, file="./final version/gbm_train.RData")


### Make prediction 
tm_test_gbm <- system.time(gbm_test <- test_gbm(gbm_train, data_test))
save(gbm_test, file="./output/gbm_test.RData")
1-sum(gbm_test!=lab_test)/666


### CV
n <- 2000
P <- 5

assign_chk <- sample(rep(1:P, times=1000/P))
assign_dog <- sample(rep(1:P, times=1000/P))
CV_index <- c(assign_chk, assign_dog)

train_time <- rep(0, P)
CV_fit_baseline <- rep(0, n)

for (c in 1:P){
  cat("fold= ", c, "\n")
  ind_test <- which(CV_index == c)
  
  data_train <- X[-ind_test,]
  lab_train <- y[-ind_test]
  data_test <- X[ind_test,]
  lab_test <- y[ind_test]
  
  train_time[c] <- system.time(mod_train <- train_gbm(data_train, lab_train, par_best))[1]
  pred_test <- test_gbm(mod_train, data_test)
  CV_fit_baseline[ind_test] <- pred_test
}
CV_err_rate <- mean(CV_fit_baseline != y)

save(train_time, CV_fit_baseline, CV_err_rate, file="./output/gbm_cv_test.RData")

#######################
####### SVM ###########
#######################



train_svm <- function(dat_train, label_train, cost = 100, gamma = 1){
  
  library("e1071")
  
  fit_svm <- svm(x=dat_train, y=label_train, cost = cost, gamma = gamma)
  
  return(fit_svm)
}

test_svm <- function(fit_train, dat_test){

  library("e1071")
  
  pred <- predict(fit_train, newdata=dat_test)
  
  return(as.numeric(pred > 0.5))
}

### Model selection with cross-validation
cv_svm <- function(X.train, y.train, K, train, test, C = 100, G = 1){
  
  n <- length(y.train)
  n.fold <- floor(n/K)
  s <- sample(rep(1:K, c(rep(n.fold, K-1), n-(K-1)*n.fold)))  
  cv.error <- rep(NA, K)
  
  for (i in 1:K){
    train.data <- X.train[s != i,]
    train.label <- y.train[s != i]
    test.data <- X.train[s == i,]
    test.label <- y.train[s == i]
    
    fit <- train(train.data, train.label, cost = C, gamma = G)
    pred <- test(fit, test.data)  
    cv.error[i] <- mean(pred != test.label)  
    
  }			
  return(c(mean(cv.error),sd(cv.error)))
  
}



# Choosing between different values of cost C for SVM
C <- c(1, 10, 100, 250, 500, 750, 1000)
err_cv_svm_c <- array(dim=c(length(C), 2))
K <- 5  # number of CV folds
for(k in 1:length(C)){
  cat("k=", k, "\n")
  err_cv_svm_c[k,] <- cv_svm(X, y, K, train_svm, test_svm, C = C[k])
}
save(err_cv_svm_c, file="./output/svm_c_err_cv.RData")

# Choose the best parameter value
C_best <- C[which.min(err_cv_svm_c[,1])]


# Choosing between different values of gamma G for SVM
G <- c(1, 5, 10, 20, 50)
err_cv_svm_g <- array(dim=c(length(G), 2))
K <- 5  # number of CV folds
for(k in 1:length(G)){
  cat("k=", k, "\n")
  err_cv_svm_g[k,] <- cv_svm(X, y, K, train_svm, test_svm, C = C_best, G = G[k])
}
save(err_cv_svm_g, file="./output/svm_g_err_cv.RData")

# Choose the best parameter value
G_best <- G[which.min(err_cv_svm_g[,1])]


########tunning use tune function
tune.svm(X, y, gamma = )



# train the model with the entire training set
tm_train_svm <- system.time(svm_train <- train_svm(data_train, lab_train))
save(svm_train, file="./final version/svm_train.RData")
tm_train_svm[1]

### Make prediction 
tm_test_svm <- system.time(svm_test <- test_svm(svm_train, data_test))
save(svm_test, file="./output/svm_test.RData")
tm_test_svm[1]
1-sum(svm_test!=lab_test)/666

### CV
train_time_svm <- rep(0, P)
CV_fit_svm <- rep(0, n)

for (c in 1:P){
  cat("fold= ", c, "\n")
  ind_test <- which(CV_index == c)
  
  data_train <- X[-ind_test,]
  lab_train <- y[-ind_test]
  data_test <- X[ind_test,]
  lab_test <- y[ind_test]
  
  train_time_svm[c] <- system.time(mod_train <- train_svm(data_train, lab_train, 
                                   cost = C_best, gamma = G_best))[1]
  pred_test <- test_svm(mod_train, data_test)
  CV_fit_svm[ind_test] <- pred_test
}
CV_err_rate_svm <- mean(CV_fit_svm != y)

save(train_time_svm, CV_fit_svm, CV_err_rate_svm, file="./output/svm_cv_test.RData")
