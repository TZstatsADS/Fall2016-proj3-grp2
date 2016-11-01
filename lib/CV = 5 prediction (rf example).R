##CV model evaluation (random forest as an example,)

source("./lib/train_rf.R")
source("./lib/test_rf.R")

k <- 5
tm_train_rf <- rep(NA, k)
tm_test_rf <- rep(NA, k)
pred_rate_rf <- rep(NA, k)

for (i in 1 : k) {
  set.seed(123 + i)
  index <- c(1:2000)
  testindex <- sample(index, 400)
  
  data_train <- X[-testindex, ]
  lab_train <- y[-testindex]
  
  data_test <- X[testindex, ]
  lab_test <- y[testindex]
  
  tm_train_rf <- system.time(rf_train <- train_rf(data_train, lab_train, 200, 2000))
  tm_train_rf[i] <- tm_train_rf[1]
  tm_test_rf <- system.time(rf_test <- test_rf(rf_train, data_test))
  tm_test_rf[i] <- tm_test_rf[1]
  
  pred_rate_rf[i] <- 1-sum(rf_test!=lab_test)/400
}
pred_rate_rf_ave <- mean(pred_rate_rf)
pred_rate_rf_ave
tm_test_rf
save(rf_train, file="./output/rf_train.RData")