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

# impute missing values, for factor variables use mode, for numeric variables use mean
## encode the imputed results as numerical
imp_train <- impute(train, classes = list(factor = imputeMode(), integer = imputeMean()), 
                    dummy.classes = c("integer","factor"), dummy.type = "numeric")
imp_train <- imp_train$data

imp_test <- impute(test, classes = list(factor = imputeMode(), integer = imputeMean()), 
                    dummy.classes = c("integer","factor"), dummy.type = "numeric")
imp_test <- imp_test$data

summary(imp_train)
summary(train)

summary(test)
summary(imp_test)

# In test data Married has no NA, but in train data it does, so in imp_train, it has Married.dummy
## In imp_train, Married NAs have been repalced, so I can simply remove Married.dummy
imp_train <- subset(imp_train, select = -c(Married.dummy))
summary(imp_train)

## !!! List algorithms that can handle missing values themselves if you don't want to deal with missing data
listLearners("classif", check.packages = TRUE, properties = "missings")[c("class","package")]

# FEATURE ENGINEERING - Data Transfer
## deal with outliers
boxplot(imp_train$ApplicantIncome)
cd <- capLargeValues(imp_train, target = "Loan_Status",cols = c("ApplicantIncome"),threshold = 40000)
hist(cd$ApplicantIncome, xlab = "ApplicantIncome", breaks = 300, main = "Applicant Income" )
boxplot(cd$ApplicantIncome)

boxplot(imp_train$CoapplicantIncome)
cd <- capLargeValues(cd, target = "Loan_Status",cols = c("CoapplicantIncome"),threshold = 20000)
boxplot(cd$CoapplicantIncome)
hist(cd$CoapplicantIncome, xlab = "Coapplicant Income", breaks = 50, main = "Coapplicant Income")

boxplot(imp_train$LoanAmount)
cd <- capLargeValues(cd, target = "Loan_Status",cols = c("LoanAmount"),threshold = 500)
boxplot(cd$LoanAmount)
hist(cd$LoanAmount, xlab = "Loan Amount", breaks = 10, main = "Loan Amount")

imp_train <- cd

imp_test$Loan_Status <- sample(0:1,size = 367,replace = T)

boxplot(imp_test$ApplicantIncome)
cd <- capLargeValues(imp_test, target = "Loan_Status",cols = c("ApplicantIncome"),threshold = 30000)
hist(cd$ApplicantIncome, xlab = "ApplicantIncome", breaks = 300, main = "Applicant Income" )
boxplot(cd$ApplicantIncome)

boxplot(imp_test$CoapplicantIncome)
cd <- capLargeValues(cd, target = "Loan_Status",cols = c("CoapplicantIncome"),threshold = 10000)
boxplot(cd$CoapplicantIncome)
hist(cd$CoapplicantIncome, xlab = "Coapplicant Income", breaks = 50, main = "Coapplicant Income")

boxplot(imp_test$LoanAmount)
cd <- capLargeValues(cd, target = "Loan_Status",cols = c("LoanAmount"),threshold = 350)
boxplot(cd$LoanAmount)
hist(cd$LoanAmount, xlab = "Loan Amount", breaks = 10, main = "Loan Amount")

imp_test <- cd

summary(imp_train)
summary(imp_test)


## convetr dummry variables fromnumeric to factor
summarizeColumns(imp_train)
for (f in names(imp_train[, c(14:19)])) {
  if( class(imp_train[, c(14:19)] [[f]]) == "numeric"){
    levels <- unique(imp_train[, c(14:19)][[f]])
    imp_train[, c(14:19)][[f]] <- as.factor(factor(imp_train[, c(14:19)][[f]], levels = levels))
  }
}
summarizeColumns(imp_train)

summarizeColumns(imp_test)
for (f in names(imp_test[, c(13:18)])) {
  if( class(imp_test[, c(13:18)] [[f]]) == "numeric"){
    levels <- unique(imp_test[, c(13:18)][[f]])
    imp_test[, c(13:18)][[f]] <- as.factor(factor(imp_test[, c(13:18)][[f]], levels = levels))
  }
}
summarizeColumns(imp_test)


# FEATURE ENGINEERING - Create New Features
imp_train$Total_Income <- imp_train$ApplicantIncome + imp_train$CoapplicantIncome
imp_test$Total_Income <- imp_test$ApplicantIncome + imp_test$CoapplicantIncome

imp_train$Income_by_Term <- imp_train$Total_Income/imp_train$LoanAmount
imp_test$Income_by_Term <- imp_test$Total_Income/imp_test$LoanAmount

imp_train$LoanAmount_by_Term <- imp_train$LoanAmount/imp_train$Loan_Amount_Term
imp_test$LoanAmount_by_Term <- imp_test$LoanAmount/imp_test$Loan_Amount_Term

summary(imp_train)
summary(imp_test)

#!!! After creating the new NUMERIC features, check their correlation with existing features
## if there is high correlation, remove the new feature since it does not add any value

## split data based on class
class_split_train <- split(names(imp_train), sapply(imp_train, function(x){ class(x)}))
class_split_train
## Total_Income and ApplicantIncome have high correlation
cor(imp_train[class_split_train$numeric])

imp_train$Total_Income <- NULL
imp_test$Total_Income <- NULL
imp_test$Loan_Status <- as.factor(imp_test$Loan_Status)
levels(imp_test$Loan_Status)[1] <- "N"
levels(imp_test$Loan_Status)[2] <- "Y"


# a task here is the dataset for learning, set positive class as "Y"
trainTask <- makeClassifTask(data = imp_train,target = "Loan_Status", positive = "Y")
trainTask
str(getTaskData(trainTask))
testTask <- makeClassifTask(data = imp_test,target = "Loan_Status", positive = "Y")
testTask
str(getTaskData(testTask))

# normalized skewed variables
trainTask <- normalizeFeatures(trainTask,method = "standardize")
summary(getTaskData((trainTask)))
testTask <- normalizeFeatures(testTask,method = "standardize")
summary(getTaskData((testTask)))

# drop unnecessary features
trainTask <- dropFeatures(task = trainTask,features = c("Loan_ID"))
testTask <- dropFeatures(task = testTask,features = c("Loan_ID"))

# feature importance
im_feat <- generateFilterValuesData(trainTask, method = c("information.gain","chi.squared", "rf.importance"))
plotFilterValues(im_feat,n.show = 20)

# !! This feature is awesome!
plotFilterValuesGGVIS(im_feat)


# QDA, a parametric algorithm, when the data follows the assumption, this type of algorithm works well
qda_learner <- makeLearner("classif.qda", predict.type = "response")
qda_model <- train(qda_learner, trainTask)
qpredict <- predict(qda_model, testTask)
qpredict


# Logistic Regression
lr_learner <- makeLearner("classif.logreg",predict.type = "response")
## cross validation
cv_lr <- crossval(learner = lr_learner,task = trainTask,iters = 5,stratify = TRUE,measures = acc,show.info = F)
cv_lr
cv_lr$aggr   ## average accuracy
cv_lr$measures.test  ## accuracy in each fold

lr_model <- train(lr_learner,trainTask)
getLearnerModel(lr_model)
lrpredict <- predict(lr_model, testTask)
lrpredict


# desicion tree, capture non-linear relations better than a logistic regression
## list all the tunable params
getParamSet("classif.rpart")

dt_learner <- makeLearner("classif.rpart", predict.type = "response")
cv_dt <- makeResampleDesc("CV",iters = 5L)
## grid search and param tuning
gs <- makeParamSet(
  makeIntegerParam("minsplit",lower = 10, upper = 50),
  makeIntegerParam("minbucket", lower = 5, upper = 50),
  makeNumericParam("cp", lower = 0.001, upper = 0.2)
)
gscontrol <- makeTuneControlGrid()
stune <- tuneParams(learner = dt_learner, resampling = cv_dt, task = trainTask, par.set = gs, control = gscontrol, measures = acc)
stune$x ## best params
stune$y ## cv accuracy

t.tree <- setHyperPars(dt_learner, par.vals = stune$x)
dt_model <- train(t.tree, trainTask)
getLearnerModel(dt_model)
dtpredict <- predict(dt_model, testTask)
dtpredict
