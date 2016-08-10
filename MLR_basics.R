library(mlr)

# list its built-in algorithms
listLearners("classif")[c("class","package")]   # classification
listLearners("regr")[c("class","package")]      # regression
listLearners("surv")[c("class","package")]      # surival
listLearners("costsens")[c("class","package")]  # cost sensitive classification
listLearners("cluster")[c("class","package")]   # clustering
listLearners("multilabel")[c("class","package")]  # multi-classification

#set working directory
path<- "[your file path]"
setwd(path)

train <- read.csv("MLR_train.csv", na.strings = c("", " ", NA))
test <- read.csv("MLR_test.csv", na.strings = c("", " ", NA))
