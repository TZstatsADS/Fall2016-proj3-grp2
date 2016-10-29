#-------------Import Data and Source Scripts-----------#
setwd("~/Desktop/Image Detection/Project3_poodleKFC_train")
source("LDA.Function.R")
setwd("~/Desktop/Image Detection/Project3_poodleKFC_train/images")

#-----Set Train and Test
set.seed(1)
index = 1:1000
Train.index = sample(1:1000,950)
Test.index = index[-Train.index]

#-----Load Texture Features---# #Not Fit using LDA
load("Dog.TF.RData")
load("Chicken.TF.RData")
Train = cbind(Chicken.TF[,Train.index],Dog.TF[,Train.index])
Test = cbind(Chicken.TF[,Test.index],Dog.TF[,Test.index])


#----Load Shape Features----#
load("Data.128.RData")
Train<-cbind(Data$chicken[,Train.index],Data$dog[,Train.index])
Test<-cbind(Data$chicke[,Test.index],Data$dog[,Test.index])
rm(Data)


#-----SIFT---------------#
library(data.table)
Features<-fread("sift_features.csv")
Features = as.matrix(Features)
set.seed(1)
index = 1:1000
Train.index = sample(index,900)
Test.index = index[-Train.index]
Train = Features[,c(Train.index,Train.index+1000)]
Test = Features[,c(Test.index,Test.index+1000)]

#-----Main Function for LDA-------#

Main.HLF.LDA<-function(Train,Test,EA,Filter = TRUE, Method = "LDA"){
n = dim(Test)[2]
True.Class = c(rep(0,n/2),rep(1,n/2))
if(Filter == TRUE){
#Filter Mask
Train.Mask = Filter.Mask(Train)
Test.Mask = Filter.Mask(Test)
}else{
  Train.Mask = Train
  Test.Mask = Test
}
rm(Test)
rm(Train)
#Two classification methods
if(Method == "PCA"){
Result = HLF.PCA(Train.Mask,Test.Mask,EA)
}else{Result = HLF.LDA(Train.Mask,Test.Mask,EA)}
#Prediction Rate
return(length(which(Result==True.Class))/length(True.Class))
}


