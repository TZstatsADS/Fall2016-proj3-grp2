Get.Texture.Features<-function(num,Name){
  library(EBImage)
  Features = NULL
  for(i in num){
    print(i)
    if(i %in% 1:9) file.name = paste0(Name,"_000",i,".jpg")
    if(i %in% 10:99) file.name = paste0(Name,"_00",i,".jpg")
    if(i %in% 100:999) file.name = paste0(Name,"_0",i,".jpg")
    if(i %in% 1000:9999) file.name = paste0(Name,"_",i,".jpg")
    Img = channel(readImage(file.name),"gray")
#---Analyze Texture Features---#
    Temp = Get.Main(Img,NP=32)
    Features = cbind(Features,t(Temp))
  }
  colnames(Features) = c(rep(Name,length(num)))
  return(Features)
}

#---Extract Features---#

Chicken.TF = Get.Texture.Features(1:1000,"chicken")
Dog.TF = Get.Texture.Features(1:1000,"dog")
save(Dog.TF,file="Dog.TF.RData")
save(Chicken.TF,file = "Chicken.TF.RData")



