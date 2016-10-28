Get.Main<-function(Img,NP = 32){
library(radiomics)
library(EBImage)
#---------Get Main Object Areas------#
Test = Img@.Data
nrow = dim(Test)[1]
ncol = dim(Test)[2]
H = hist(Test)
index = which(H$density>1)
if(length(index)==1){
  Temp1 = as.vector(Test)
  Layer = which(Temp1>=H$breaks[index] & Temp1<=H$breaks[index+1])
  Temp2 = Temp1[-Layer]
  H = hist(Temp2)
  index = which(H$density>1)
}
MainColors = H$breaks[c(min(index),max(index))]
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
col = col.min:col.max
row = row.min:row.max
if(col.min == 1){ col = 1:(NP*2)}
if(col.max == ncol){ col = (ncol-2*NP+1):ncol}
if(row.min == 1){ row = 1:(NP*2)}
if(row.max == nrow){ row = (nrow-2*NP+1):nrow}
if(max(dim(Test)<64)){ Main = Test}else{
Main = Test[row,col]
}
#---Try another part of Main---# #The alternative part works! This is weird!
#Main = Test[1:10,2:15]
#----Get Texture Features ----#
Texture = tryCatch(glcm(Main),error=function(e){return(glcm(Test))})
TF = calc_features(Texture)
Texture.Features = matrix(TF,ncol=1)
return(Texture.Features)
}

