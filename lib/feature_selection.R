setwd("~/GitHub/Fall2016-proj3-grp2")
feature <- read.csv("sift_features.csv")
X <- t(feature)
y <- rep(c(1, 0), each = 1000)
index <- c(1:2000)
testindex <- sample(index, trunc(2000/3))
lab_train <- y[-testindex]
lab_test <- y[testindex]

ft.pca <- prcomp(X)
cu.var <- summary(ft.pca)$importance[3,]

cu.var.cutoff <- seq(0.6, 0.95, by = 0.05)
l.cutoff <- length(cu.var.cutoff)
pca.rate <- rep(NA, l.cutoff)


traintest <- function(train, test, data_train, lab_train, data_test, lab_test){
  train.re <- train(data_train, lab_train)
  test.re <- test(train.re, data_test)

  return(1-sum(test.re != lab_test)/length(lab_test))
}



for (i in 1:l.cutoff){
  cat("i = ", i, "\n")
  cutoff <- cu.var.cutoff[i]
  npc <- min(which(cu.var > cutoff))
  X.pca <- as.data.frame(ft.pca$x)[,1:npc]
  
  data_train <- X.pca[-testindex, ]
  data_test <- X.pca[testindex, ]
  
  
  ##use traintest() to train and test your model under different pca cutoff
  ##change train_svm, test_svm below
  pca.rate[i] <- traintest(train_svm, test_svm, data_train, lab_train, data_test, lab_test)
}

cutoff_best <- cu.var.cutoff[which.min(pca.rate)]

