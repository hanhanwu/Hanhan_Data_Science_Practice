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

# MLR summary, gives data type too
# The mean and median has very large differences in ApplicationIncome, CoapplicantINcome, 
## this is the sign of being highly skewed, need normalization
# The min/max and the mean has huge differences in ApplicationIncome, CoapplicantINcome and LoanAmount,
## they may have outliers
summarizeColumns(train)
summarizeColumns(test)

hist(train$ApplicantIncome, xlab = "ApplicantIncome", breaks = 300, main = "Applicant Income" )
hist(train$CoapplicantIncome, xlab = "CoapplicantIncome", breaks = 50, main = "Coapplicant Income")
boxplot(train$LoanAmount)
boxplot(train$ApplicantIncome)
boxplot(train$CoapplicantIncome)

train$Credit_History <- as.factor(train$Credit_History)
class(train$Credit_History)
test$Credit_History <- as.factor(test$Credit_History)

summary(train)
levels(train$Credit_History)
levels(train$Credit_History)[1] <- 'N'
levels(train$Credit_History)[2] <- 'Y'
levels(train$Credit_History)

levels(test$Credit_History)[1] <- 'N'
levels(test$Credit_History)[2] <- 'Y'

# TO BE CONTINUED...
