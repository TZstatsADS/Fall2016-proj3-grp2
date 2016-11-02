###Train GBM model with different parameters

train_gbm <- function(dat_train, label_train, par){
  
  #remove 0 variation columns
  zero.var.col <- which(apply(dat_train, 2, var)==0)
  x <- dat_train[,-zero.var.col]
  
  #fit gbm model with selected parameters
  fit_gbm <- gbm.fit(x=x, y=label_train,
                     n.trees=2000,
                     distribution="bernoulli",
                     interaction.depth = par$depth, 
                     n.minobsinnode = par$n.minobsinnode, 
                     bag.fraction = 0.5,
                     verbose=FALSE)
  
  best_iter <- gbm.perf(fit_gbm, method = "OOB")
  
  return(list(fit=fit_gbm, iter=best_iter))
}

