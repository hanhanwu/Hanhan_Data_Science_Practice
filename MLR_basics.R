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

# Random Forest
getParamSet("classif.randomForest")
rf_learner <- makeLearner("classif.randomForest", predict.type = "response", par.vals = list(ntree = 200, mtry = 3))
rf_learner$par.vals <- list(
  importance = TRUE
)
rf_param <- makeParamSet(
  makeIntegerParam("ntree",lower = 50, upper = 500),
  makeIntegerParam("mtry", lower = 3, upper = 10),
  makeIntegerParam("nodesize", lower = 10, upper = 50)
)
## random search is faster than grid search, but may miss the real optimal
rancontrol <- makeTuneControlRandom(maxit = 50L)
cv_rf <- makeResampleDesc("CV",iters = 3L)
rf_tune <- tuneParams(learner = rf_learner, resampling = cv_rf, task = trainTask, par.set = rf_param, control = rancontrol, measures = acc)
rf_tune$x
rf_tune$y
rf.tree <- setHyperPars(rf_learner, par.vals = rf_tune$x)
rf_model <- mlr::train(learner=rf.tree, task=trainTask)
getLearnerModel(rf_model)
rfpredict <- predict(rf_model, testTask)
rfpredict


# SVM
getParamSet("classif.ksvm")
svm_learner <- makeLearner("classif.ksvm", predict.type = "response")
svm_param <- makeParamSet(
  makeDiscreteParam("C", values = 2^c(-8,-4,-2,0)), #cost parameters
  makeDiscreteParam("sigma", values = 2^c(-8,-4,0,4)) #RBF Kernel Parameter
)
## search function
ctrl <- makeTuneControlGrid()
cv_svm <- makeResampleDesc("CV",iters = 3L)
svm_tune <- tuneParams(svm_learner, task = trainTask, resampling = cv_svm, par.set = svm_param, control = ctrl,measures = acc)
svm_tune$x
svm_tune$y
t.svm <- setHyperPars(svm_learner, par.vals = svm_tune$x)
svm_model <- train(svm_learner, trainTask)
svmpredict <- predict(svm_model, testTask)
svmpredict


# GBM - after 1 round prediction, ti checks for incorrect predictions and give them weights,
## predicting them again till they are predicted correctly
getParamSet("classif.gbm")
gbm_learned <- makeLearner("classif.gbm", predict.type = "response")
rancontrol <- makeTuneControlRandom(maxit = 50L)
cv_gbm <- makeResampleDesc("CV",iters = 3L)
gbm_param<- makeParamSet(
  makeDiscreteParam("distribution", values = "bernoulli"),
  makeIntegerParam("n.trees", lower = 100, upper = 1000), #number of trees
  makeIntegerParam("interaction.depth", lower = 2, upper = 10), #depth of tree
  makeIntegerParam("n.minobsinnode", lower = 10, upper = 80),
  makeNumericParam("shrinkage",lower = 0.01, upper = 1)
)
gbm_tune <- tuneParams(learner = gbm_learned, task = trainTask,resampling = cv_gbm,measures = acc,par.set = gbm_param,control = rancontrol)
gbm_tune$x
gbm_tune$y
final_gbm <- setHyperPars(learner = gbm_learned, par.vals = gbm_tune$x)
gbm_model <- train(final_gbm, trainTask)
gbmpredict <- predict(gbm_model, testTask)


## XGboost - cannot predict well on unseen data, so if the test data has no truth, XGboost may not be a good choice
set.seed(410)
getParamSet("classif.xgboost")
xg_learner <- makeLearner("classif.xgboost", predict.type = "response")
xg_learner$par.vals <- list(
  objective = "binary:logistic",
  eval_metric = "error",
  nrounds = 250
)

xg_param <- makeParamSet(
  makeIntegerParam("nrounds",lower=200,upper=600),
  makeIntegerParam("max_depth",lower=3,upper=20),
  makeNumericParam("lambda",lower=0.55,upper=0.60),
  makeNumericParam("eta", lower = 0.001, upper = 0.5),
  makeNumericParam("subsample", lower = 0.10, upper = 0.80),
  makeNumericParam("min_child_weight",lower=1,upper=5),
  makeNumericParam("colsample_bytree",lower = 0.2,upper = 0.8)
)
rancontrol <- makeTuneControlRandom(maxit = 100L)
cv_xg <- makeResampleDesc("CV",iters = 3L)
xg_tune <- tuneParams(learner = xg_learner, task = trainTask, resampling = cv_xg,measures = acc,par.set = xg_param, control = rancontrol)
xg_final <- setHyperPars(learner = xg_learner, par.vals = xg_tune$x)
xgmodel <- train(xg_final, trainTask)
xgpredict <- predict(xgmodel, testTask)


# random forest with top 6 features
## so far random forest is the best model here, train the data with top 6 features with random forest
getParamSet("classif.randomForest")
top_task <- filterFeatures(trainTask, method = "rf.importance", abs = 6)

rf_learner <- makeLearner("classif.randomForest", predict.type = "response", par.vals = list(ntree = 200, mtry = 3))
rf_learner$par.vals <- list(
  importance = TRUE
)
rf_param <- makeParamSet(
  makeIntegerParam("ntree",lower = 50, upper = 500),
  makeIntegerParam("mtry", lower = 3, upper = 10),
  makeIntegerParam("nodesize", lower = 10, upper = 50)
)
## random search is faster than grid search, but may miss the real optimal
rancontrol <- makeTuneControlRandom(maxit = 50L)
cv_rf <- makeResampleDesc("CV",iters = 3L)
rf_tune <- tuneParams(learner = rf_learner, resampling = cv_rf, task = top_task, par.set = rf_param, control = rancontrol, measures = acc)
rf_tune$x
rf_tune$y
rf.tree <- setHyperPars(rf_learner, par.vals = rf_tune$x)
rf_model <- train(rf.tree, top_task)
getLearnerModel(t.rpart)
rfpredict <- predict(rf_model, testTask)
rfpredict
result <- data.frame(test$Loan_ID, rfpredict$data$response)
