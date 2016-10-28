Get.Texture.Features<-function(num,Name){
  library(EBImage)
  Features = NULL
  for(i in num){
    print(i)
    if(i %in% 1:9) file.name = paste0(Name,"_000",i,".jpg")
    if(i %in% 10:99) file.name = paste0(Name,"_00",i,".jpg")
    if(i %in% 100:999) file.name = paste0(Name,"_0",i,".jpg")
    if(i %in% 1000:9999) file.name = paste0(Name,"_",i,".jpg")
    A = channel(readImage(file.name),"gray")
    Temp = Get.Main(A)
    Features = cbind(Features,Temp)
  }
  colnames(Features) = c(rep(Name,length(num)))
  return(Features)
}

#A = Get.Texture.Features(1:5,"chicken")

