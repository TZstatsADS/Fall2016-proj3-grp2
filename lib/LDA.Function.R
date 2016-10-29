#------Import Images into Grey Scale ------#
read.image.grey<-function(num,size){
  library(EBImage)
  img.chicken = NULL
  for(i in num){
    print(i)
    if(i %in% 1:9) file.name = paste0("chicken_000",i,".jpg")
    if(i %in% 10:99) file.name = paste0("chicken_00",i,".jpg")
    if(i %in% 100:999) file.name = paste0("chicken_0",i,".jpg")
    if(i %in% 1000:9999) file.name = paste0("chicken_",i,".jpg")
    A = channel(readImage(file.name),"gray")
    img1 = as.vector(resize(channel(readImage(file.name),"gray"),w=size,h=size))
    img.chicken = cbind(img.chicken,img1)
  }
  img.dog = NULL
  for(i in num){
    print(i)
    if(i %in% 1:9) file.name = paste0("dog_000",i,".jpg")
    if(i %in% 10:99) file.name = paste0("dog_00",i,".jpg")
    if(i %in% 100:999) file.name = paste0("dog_0",i,".jpg")
    if(i %in% 1000:9999) file.name = paste0("dog_",i,".jpg")
    img1 = as.vector(resize((channel(readImage(file.name),"gray")),w =size, h = size))
    img.dog = cbind(img.dog,img1)
  }
  return(list(chicken = img.chicken, dog = img.dog))
}


#-------Use PCA to reduce dimensions--------------#
PCA.image<-function(IMatrix,EA){
  # Imatrix.pca = prcomp(Imatrix,
  #                      center = FALSE,
  #                      scale. = TRUE)
  sv<-svd(IMatrix)
  D = sv$d
  PV = cumsum(D^2)/sum(D^2)
  depth = which(PV>=EA)[1]
  D = diag(D)[1:depth,1:depth]
  KLBasis = sv$u[,1:depth]
  V = sv$v[,1:depth]
  A = D%*%t(V)
  return(list(KLBasis = KLBasis, PCA.A = A))
}

#------Use Mean Values to filter Image--------#

Filter.Mask<-function(Image){
  for(i in 1:dim(Image)[2]){
    temp = Image[,i]
    m = median(temp)
    index0<-which(temp<m)
    Image[,i]=1
    Image[index0,i] = 0
  }
  return(Filter.Image = Image)
}

#--------HLF.LDA.Function----#

HLF.LDA<-function(Train,Test,EA,PCA){
  #reduce dimensions
  Train<-PCA.image(Train,EA)
  PCA.Train = Train$PCA.A
  n = dim(PCA.Train)[2]
  Chicken = PCA.Train[,1:(n/2)]
  Dog = PCA.Train[,((n/2+1):n)]  
  Test = t(Train$KLBasis)%*%Test
  #Perform LDA
  W = LDA(Chicken,Dog)
  C = t(W)%*%Chicken
  D = t(W)%*%Dog
  #Find threshold value alpha
  Cmin = min(C)
  Cmax = max(C)
  Dmin = min(D)
  Dmax = max(D)
  Cpos = 0
  if(Dmax<Cmax){#Chicken are left of alpha
    alpha = (Dmax + Cmin)/2
    Cpos =1
  } else{
    alpha = (Dmin+Cmax)/2
  }
  result = numeric(dim(Test)[2])
  #Classify
  for(i in 1:dim(Test)[2]){
    P = t(W)%*%Test[,i]
    if(Cpos == 1){
      if(P<alpha) result[i] = 1 #Chicken
    }else{
      if(P>alpha) result[i] = 1
    }
  }
  return(result)
}


#---------Basic LDA-----#

LDA<-function(Chicken,Dog){
  library(geigen)
  nc = dim(Chicken)[2]
  nd = dim(Dog)[2]
  M = dim(Dog)[1]
  mc = apply(Chicken,1,mean)
  md = apply(Dog,1,mean)
  v = mc-md #chicken - dog
  #Between-class
  Sb = v%*%t(v)
  #Within-class
  Sw = matrix(0,M,M)
  for(i in 1:nc){
    v = Chicken[,i]-mc
    Sw = Sw + v%*%t(v)
  }
  for(i in 1:nd){
    v = Dog[,i] - md
    Sw = Sw + v%*%t(v)
  }
  #Solve generalized eigenvalue problem
  ge = geigen(Sb,Sw)
  index = which.max(ge$values)
  V = ge$vectors
  W = V[,index]/norm(as.matrix(V[,index]),"2")
  return(W) #Optimal Projection vector
}


#---------HLF.PCA------------------#
HLF.PCA<-function(Train,Test,EA){
  Result = numeric(dim(Test)[2])
  n = dim(Train)[2]
  Set0 = Train[,(1:(n/2))]
  Set1 = Train[,((n/2+1):n)]
  #Construct Optimal Bases
  Chicken = PCA.image(Set0,EA)
  Dog = PCA.image(Set1,EA)
  #Novelty of test data
  for(i in 1:dim(Test)[2]){
    x0 = t(Chicken$KLBasis)%*%Test[,i] #Chicken 0
    x1 = t(Dog$KLBasis)%*%Test[,i] #Dog 1
    n0 = norm(Chicken$KLBasis%*%x0-Test[,i],"2")
    n1 = norm(Dog$KLBasis%*%x1 - Test[,i],"2") 
    if( n1 < n0) Result[i] = 1
  }
  return(Result)
}






