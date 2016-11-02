#Test with SVM model

test_svm <- function(fit_train, dat_test){

  pred <- predict(fit_train, newdata=dat_test)
  
  return(as.numeric(pred > 0.5))
}