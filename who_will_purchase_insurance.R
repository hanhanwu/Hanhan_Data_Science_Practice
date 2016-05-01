#set working directory
path<- "/Users/hanhanwu/Desktop"
setwd(path)

# load data, 298 features + 1 column of label
train <- read.csv("insurance_train.csv")
test <- read.csv("insurance_test.csv")

# quick explore data
dim(train)
dim(test)
str(train)
str(test)
summary(train)
# check missisng data
table(is.na(train))
colSums(is.na(train))

# create combi
test$QuoteConversion_Flag <- 1
combi <- rbind(train, test)

# PropertyField29 has too many missing values in training data, remove it
# PersonalField84 has around half missing values in training data, remove it
combi <- subset(combi, select = -c(PropertyField29, PersonalField84))

table(is.na(combi))

# remove indicator and identifier columns
combi <- subset(combi, select = -c(QuoteConversion_Flag, QuoteNumber))


# using FADM for dimensional reduction
# FAMD is a principal component method dedicated to explore data 
## with both continuous and categorical variables. 
## It can be seen roughly as a mixed between PCA and MCA. 
library(FactoMineR)
res <- FAMD(combi, ncp=20)
# runing out of memory on my laptop


# using PCA and convert categorical data into numerical data first
summary(combi)
levels(combi$GeographicField63)
levels(combi$GeographicField63)[1] <- "Other"
levels(combi$GeographicField63)

levels(combi$PropertyField38)
levels(combi$PropertyField38)[1] <- "Other"
levels(combi$PropertyField38)

levels(combi$PropertyField37)
levels(combi$PropertyField37)[3] <- "Other"
levels(combi$PropertyField37)

levels(combi$PropertyField36)
levels(combi$PropertyField36)[1] <- "Other"
levels(combi$PropertyField36)

levels(combi$PropertyField34)
levels(combi$PropertyField34)[1] <- "Other"
levels(combi$PropertyField34)

levels(combi$PropertyField32)
levels(combi$PropertyField32)[1] <- "Other"
levels(combi$PropertyField32)

levels(combi$PropertyField30)
levels(combi$PropertyField30)[3] <- "Other"
levels(combi$PropertyField30)

levels(combi$PropertyField4)
levels(combi$PropertyField4)[1] <- "Other"
levels(combi$PropertyField4)

levels(combi$PropertyField5)
levels(combi$PropertyField5)[3] <- "Other"
levels(combi$PropertyField5)

levels(combi$PropertyField3)
levels(combi$PropertyField3)[1] <- "Other"
levels(combi$PropertyField3)


levels(combi$PersonalField7)
levels(combi$PersonalField7)[1] <- "Other"
levels(combi$PersonalField7)

#convert categorical data into numerical data
library(dummies)

combi$GeographicField63[is.na(combi$GeographicField63)] <- "N"

combi$GeographicField64 <- as.numeric(factor(combi$GeographicField64, 
                                        labels=(1:length(levels(factor(combi$GeographicField64))))))
combi$GeographicField63 <- as.numeric(factor(combi$GeographicField63, 
                                             labels=(1:length(levels(factor(combi$GeographicField63))))))
combi$PropertyField38 <- as.numeric(factor(combi$PropertyField38, 
                                           labels=(1:length(levels(factor(combi$PropertyField38))))))
combi$PropertyField37 <- as.numeric(factor(combi$PropertyField37, 
                                           labels=(1:length(levels(factor(combi$PropertyField37))))))
combi$PropertyField36 <- as.numeric(factor(combi$PropertyField36, 
                                       labels=(1:length(levels(factor(combi$PropertyField36))))))
combi$PropertyField28 <- as.numeric(factor(combi$PropertyField28, 
                                           labels=(1:length(levels(factor(combi$PropertyField28))))))
combi$PropertyField30 <- as.numeric(factor(combi$PropertyField30, 
                                           labels=(1:length(levels(factor(combi$PropertyField30))))))
combi$PropertyField31 <- as.numeric(factor(combi$PropertyField31, 
                                           labels=(1:length(levels(factor(combi$PropertyField31))))))
combi$PropertyField32 <- as.numeric(factor(combi$PropertyField32, 
                                           labels=(1:length(levels(factor(combi$PropertyField32))))))
combi$PropertyField33 <- as.numeric(factor(combi$PropertyField33, 
                             labels=(1:length(levels(factor(combi$PropertyField33))))))
combi$PropertyField34 <- as.numeric(factor(combi$PropertyField34, 
                           labels=(1:length(levels(factor(combi$PropertyField34))))))
combi$PropertyField14 <- as.numeric(factor(combi$PropertyField14, 
                           labels=(1:length(levels(factor(combi$PropertyField14))))))
combi$PropertyField7 <- as.numeric(factor(combi$PropertyField7, 
                                           labels=(1:length(levels(factor(combi$PropertyField7))))))
combi$PropertyField4 <- as.numeric(factor(combi$PropertyField4, 
                                          labels=(1:length(levels(factor(combi$PropertyField4))))))
combi$PropertyField5 <- as.numeric(factor(combi$PropertyField5, 
                                          labels=(1:length(levels(factor(combi$PropertyField5))))))
combi$PropertyField3 <- as.numeric(factor(combi$PropertyField3, 
                                          labels=(1:length(levels(factor(combi$PropertyField3))))))
combi$PersonalField16 <- as.numeric(factor(combi$PersonalField16, 
                                           labels=(1:length(levels(factor(combi$PersonalField16))))))
combi$PersonalField17 <- as.numeric(factor(combi$PersonalField17, 
                              labels=(1:length(levels(factor(combi$PersonalField17))))))
combi$PersonalField18 <- as.numeric(factor(combi$PersonalField18, 
                                           labels=(1:length(levels(factor(combi$PersonalField18))))))
combi$PersonalField19 <- as.numeric(factor(combi$PersonalField19, 
                                           labels=(1:length(levels(factor(combi$PersonalField19))))))
combi$PersonalField7 <- as.numeric(factor(combi$PersonalField7, 
                                          labels=(1:length(levels(factor(combi$PersonalField7))))))
combi$SalesField7 <- as.numeric(factor(combi$SalesField7, 
                                       labels=(1:length(levels(factor(combi$SalesField7))))))
combi$CoverageField8  <- as.numeric(factor(combi$CoverageField8 , 
                                           labels=(1:length(levels(factor(combi$CoverageField8))))))
combi$CoverageField9  <- as.numeric(factor(combi$CoverageField9 , 
                                           labels=(1:length(levels(factor(combi$CoverageField9))))))
combi$Original_Quote_Date <- as.numeric(factor(combi$Original_Quote_Date, 
                                               labels=(1:length(levels(factor(combi$Original_Quote_Date))))))
combi$Field12 <- as.numeric(factor(combi$Field12, 
                        labels=(1:length(levels(factor(combi$Field12))))))
combi$Field6 <- as.numeric(factor(combi$Field6, 
                                  labels=(1:length(levels(factor(combi$Field6))))))
combi$Field10 <- as.numeric(factor(combi$Field10, 
                                   labels=(1:length(levels(factor(combi$Field10))))))
str(combi)

# PCA
## check columns with 0 variance and remove them
names(combi[, sapply(combi, function(v) var(v, na.rm=TRUE)==0)])
combi <- subset(combi, select = -c(PropertyField6, GeographicField10A))
prin_comp <- prcomp(combi, scale. = T)

#compute standard deviation of each principal component
std_dev <- prin_comp$sdev
#compute variance
pr_var <- std_dev^2
#check variance of first 10 components
pr_var[1:10]
# higher is the explained variance, higher will be the information contained in those components
## proportion of variance explained
prop_varex <- pr_var/sum(pr_var)
prop_varex[1:20]

#scree plot
plot(prop_varex, xlab = "Principal Component",
     ylab = "Proportion of Variance Explained",
     type = "b")

# plot a cumulative variance plot, get give us a clear picture of number of components
plot(cumsum(prop_varex), xlab = "Principal Component",
     ylab = "Cumulative Proportion of Variance Explained",
     type = "b")
# The first 150-180 columns contain the most info


# predictive analysis
new_train <- combi[1:nrow(train),]
new_train <- subset(new_train, select = 1:150)
new_test <- combi[-(1:nrow(train)),]
new_test <- subset(new_test, select = 1:150)
new_train$QuoteConversion_Flag <- train$QuoteConversion_Flag
dim(new_train)
dim(new_test)

## SVM (too slow in R on my laptop)
library("e1071")
svm_model <- svm(QuoteConversion_Flag ~ ., data=new_train)

## NN (too slow in R on my laptop)
library(MASS)
library(neuralnet)
n <- names(new_train)
f <- as.formula(paste("QuoteConversion_Flag ~", 
                      paste(n[!n %in% "QuoteConversion_Flag"], collapse = " + ")))
hd <- length(n)
nn <- neuralnet(f,data=new_train,hidden=hd,linear.output=T)

plot(nn)

pr.nn <- compute(nn,new_test)
result$QuoteConversion_Flag <- pr.nn$net.result
result$QuoteNumber <- new_text$QuoteNumber
write.csv(result, "result.csv")

# output the processed data so that they can be used on other machine learning tools
write.csv(new_train, "insurance_new_train.csv", row.names=FALSE)
write.csv(new_test, "insurance_new_test.csv", row.names=FALSE)
