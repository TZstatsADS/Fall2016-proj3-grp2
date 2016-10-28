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
    Features = cbind(Features,Temp)
  }
  colnames(Features) = c(rep(Name,length(num)))
  return(Features)
}

#---Extract Features---#
Chicken.TF1 = Get.Texture.Features(1:180,"chicken")
Chicken.TF2 = Get.Texture.Features(182:439,"chicken")
Chicken.TF3 = Get.Texture.Features(441:797,"chicken")
Chicken.TF4 = Get.Texture.Features(799:983,"chicken")
Chicken.TF5 = Get.Texture.Features(985:1000,"chicken")

# For chicken Pictures---Below are the failed pictures--#
Name = "chicken"
i = 181
i = 440
i = 798
i = 984
#
