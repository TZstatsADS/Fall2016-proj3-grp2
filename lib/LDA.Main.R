
setwd("~/Desktop/Image Detection/Project3_poodleKFC_train")
source("LDA.Function.R")

setwd("~/Desktop/Image Detection/Project3_poodleKFC_train/images")
set.seed(1)
index = 1:1000
Train.index = sample(1:1000,950)
Test.index = index[-Train.index]
Data<-read.image.grey(index,400)
#save(Data,file = "Data.128.RData")
load("Data.128.RData")

#Using Own Features
Train<-cbind(Data$chicken[,Train.index],Data$dog[,Train.index])
Test<-cbind(Data$chicke[,Test.index],Data$dog[,Test.index])
n = dim(Test)[2]
True.Class = c(rep(0,n/2),rep(1,n/2))
#Filter Mask
Train.Mask = Filter.Mask(Train)
Test.Mask = Filter.Mask(Test)
#Two classification methods
LDA.Result = HLF.LDA(Train.Mask,Test.Mask,0.97)
PCA.Result = HLF.PCA(Train.Mask,Test.Mask,0.97)
#Prediction Rate
length(which(PCA.Result==True.Class))/length(True.Class)
length(which(LDA.Result==True.Class))/length(True.Class)



#Using SIFT Features
library(data.table)
Features<-fread("sift_features.csv")
Features = as.matrix(Features)
set.seed(1)
index = 1:1000
Train.index = sample(index,900)
Test.index = index[-Train.index]
Train = Features[,c(Train.index,Train.index+1000)]
Test = Features[,c(Test.index,Test.index+1000)]
n = dim(Test)[2]
True.Class = c(rep(0,n/2),rep(1,n/2))
LDA.Result = HLF.LDA(Train,Test,0.97)
PCA.Result = HLF.PCA(Train,Test,0.97)
#Prediction Rate
length(which(PCA.Result==True.Class))/length(True.Class)
length(which(LDA.Result==True.Class))/length(True.Class)
