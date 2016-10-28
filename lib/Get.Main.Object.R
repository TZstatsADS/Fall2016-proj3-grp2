Get.Main<-function(Img,NP = 32){
library(radiomics)
library(EBImage)
#---------Get Main Object Areas------#
Test = Img@.Data
H = hist(Test)
index = which(H$density>1)
MainColors = H$breaks[c(min(index),max(index))]
nrow = dim(Test)[1]
ncol = dim(Test)[2]
Temp = as.vector(Test)
Temp[Temp<MainColors[1] | Temp>MainColors[2]] = 0
Temp[Temp!=0] = 1
#------Select a submatrix for analyze--#
Test1 = matrix(Temp,nrow,ncol)
col.index = which.max(apply(Test1,2,sum)) 
row.index = which.max(apply(Test1,1,sum)) 
col.min = max(1,(col.index-NP)) 
col.max = min(ncol,(col.index+NP-1)) 
row.min = max(1,(row.index-NP)) 
row.max = min(nrow,(row.index+NP-1)) 
Main = Test[row.min:row.max,col.min:col.max]

#---Try another part of Main---# #The alternative part works! This is weird!
#Main = Test[1:10,2:15]

#----Get Texture Features ----#
Texture = glcm(Main)
TF = calc_features(Texture)
Texture.Features = matrix(TF,ncol=1)
return(Texture.Features)
}

