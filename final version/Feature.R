feature_sift <- read.csv("sift features_test.csv")
feature.baseline <- t(feature_sift)

feature_bow <- read.csv("bow_test.csv")
feature.adv <- feature_bow[,2:501]


save(feature.baseline, feature.adv, file = "Feature_eval.RData")


