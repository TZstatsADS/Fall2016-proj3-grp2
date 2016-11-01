setwd("~/GitHub/Fall2016-proj3-grp2")
feature <- read.csv("sift_features.csv")
X <- t(feature)
y <- rep(c(1, 0), each = 1000)
index <- c(1:2000)
testindex <- sample(index, trunc(2000/3))
lab_train <- y[-testindex]
lab_test <- y[testindex]
data_train <- X[-testindex, ]
data_test <- X[testindex, ]

ft.pca <- prcomp(data_train)
data_test_pca <- t(apply(data_test, 1, function(x) 
                    x-as.vector(ft.pca$center))) %*% 
                    as.matrix(ft.pca$rotation)

cu.var <- summary(ft.pca)$importance[3,]
cu.var.cutoff <- seq(0.6, 0.95, by = 0.05)
l.cutoff <- length(cu.var.cutoff)
pca.rate <- rep(NA, l.cutoff)


traintest <- function(train, test, data_train_pca, 
                      lab_train, data_test_pca, lab_test){
  train.re <- train(data_train, lab_train)
  test.re <- test(train.re, data_test)

  return(1-sum(test.re != lab_test)/length(lab_test))
}



for (i in 1:l.cutoff){
  cat("i = ", i, "\n")
  cutoff <- cu.var.cutoff[i]
  npc <- min(which(cu.var > cutoff))
  X.pca <- as.data.frame(ft.pca$x)[,1:npc]
  
  data_train_pca <- X.pca
  
  ##use traintest() to train and test your model under different pca cutoff
  ##change train_svm, test_svm below
  pca.rate[i] <- traintest(train_gbm, test_gbm, data_train_pca, 
                           lab_train, data_test_pca, lab_test)
}

cutoff_best <- cu.var.cutoff[which.min(pca.rate)]

