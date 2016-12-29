library(RODBC)
library(data.table)
require(data.table)
library(dplyr)
library(caret)
library(dummies)
library(ggplot2)
library(plotly)
library(FSelector)
library('e1071')
library(mlr)
library(ROSE)



# Seperate data into 2/3 and 1/3
idx <- createDataPartition(q2$HasWriteOff, p=0.67, list=FALSE)
train_data <- q2[idx,]
test_data <- q2[-idx,]

train_task <- makeClassifTask(data=data.frame(train_data), target = "HasWriteOff", positive = "2")
test_task <- makeClassifTask(data=data.frame(test_data), target = "HasWriteOff", positive = "2")


# Method 1 - Random Forest
set.seed(1)
getParamSet("classif.randomForest")
rf_learner <- makeLearner("classif.randomForest", predict.type = "response", par.vals = list(ntree = 200, mtry = 3))
rf_learner$par.vals <- list(importance = TRUE)
rf_param <- makeParamSet(
  makeIntegerParam("ntree",lower = 50, upper = 500),
  makeIntegerParam("mtry", lower = 3, upper = 10),
  makeIntegerParam("nodesize", lower = 10, upper = 50)
)
rancontrol <- makeTuneControlRandom(maxit = 50L)
cv_rf <- makeResampleDesc("CV",iters = 5L)
rf_tune <- tuneParams(learner = rf_learner, resampling = cv_rf, task = train_task, par.set = rf_param, control = rancontrol, measures = acc)
rf_tune$x
rf_tune$y
rf.tree <- setHyperPars(rf_learner, par.vals = rf_tune$x)
rf_model <- mlr::train(learner=rf.tree, task=train_task)
getLearnerModel(rf_model)
rfpredict <- predict(rf_model, test_task)
nb_prediction <- rfpredict$data$response
dCM <- confusionMatrix(test_data$HasWriteOff, nb_prediction, positive = "2")
dCM    

## check generated feature importance
rf_feature_importance <- data.frame(rf_model$learner.model$importance)
feature_names <- data.frame(colnames(train_data)[1:(length(train_data)-1)])
colnames(feature_names)[1] <- 'Feature'
fi <- cbind(feature_names, rf_feature_importance$MeanDecreaseGini)
colnames(fi)[2] <- "MeanDecreaseGini"
setorder(fi, -"MeanDecreaseGini")
head(fi, n=15)


# method 2 - XGBOOst
set.seed(1)
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
cv_xg <- makeResampleDesc("CV",iters = 5L)
xg_tune <- tuneParams(learner = xg_learner, task = train_task, resampling = cv_xg,measures = acc,par.set = xg_param, control = rancontrol)
xg_final <- setHyperPars(learner = xg_learner, par.vals = xg_tune$x)
xgmodel <- mlr::train(xg_final, train_task)
xgpredict <- predict(xgmodel, test_task)
nb_prediction <- xgpredict$data$response
dCM <- confusionMatrix(test_data$HasWriteOff, nb_prediction, positive = "2")
dCM


# method 3 - C50
getParamSet("classif.C50")
c50_learner <- makeLearner("classif.C50", predict.type = "response", par.vals = list(seed=410, noGlobalPruning=T, subset=T))
c50_param <- makeParamSet(
  makeIntegerParam("trials",lower = 10, upper = 100)
)
rancontrol <- makeTuneControlRandom(maxit = 50L)
cv_c50 <- makeResampleDesc("CV",iters = 5L)
c50_tune <- tuneParams(learner = c50_learner, resampling = cv_c50, task = train_task, par.set = c50_param, control = rancontrol, measures = acc)
c50_tune$x
c50_tune$y
c50.tree <- setHyperPars(c50_learner, par.vals = c50_tune$x)
c50_model <- mlr::train(learner=c50.tree, task=train_task)
getLearnerModel(c50_model)
c50predict <- predict(c50_model, test_task)
nb_prediction <- c50predict$data$response
dCM <- confusionMatrix(test_data$HasWriteOff, nb_prediction, positive = "2")
dCM


# method 4 - GBM
set.seed(1)
getParamSet("classif.gbm")
gbm_learner <- makeLearner("classif.gbm", predict.type = "response")
rancontrol <- makeTuneControlRandom(maxit = 50L)
cv_gbm <- makeResampleDesc("CV",iters = 3L)
gbm_param<- makeParamSet(
  makeDiscreteParam("distribution", values = "bernoulli"),
  makeIntegerParam("n.trees", lower = 100, upper = 1000), 
  makeIntegerParam("interaction.depth", lower = 2, upper = 10), 
  makeIntegerParam("n.minobsinnode", lower = 10, upper = 80),
  makeNumericParam("shrinkage",lower = 0.01, upper = 1)
)
gbm_tune <- tuneParams(learner = gbm_learner, task = train_task,resampling = cv_gbm,measures = acc,par.set = gbm_param,control = rancontrol)
gbm_tune$x
gbm_tune$y
final_gbm <- setHyperPars(learner = gbm_learner, par.vals = gbm_tune$x)
gbm_model <- mlr::train(final_gbm, train_task)
gbmpredict <- predict(gbm_model, test_task)
nb_prediction <- gbmpredict$data$response
dCM <- confusionMatrix(test_data$HasWriteOff, nb_prediction, positive = "2")
dCM


#method 5 - ROSE + XGBOOST
bal_data_rose <- ROSE(HasWriteOff~., data = train_data, seed = 1)$data
table(bal_data_rose$HasWriteOff)
train_task_rose <- makeClassifTask(data=data.frame(bal_data_rose), target = "HasWriteOff", positive = "2")
set.seed(1)
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
cv_xg <- makeResampleDesc("CV",iters = 5L)
xg_tune <- tuneParams(learner = xg_learner, task = train_task_rose, resampling = cv_xg,measures = acc,par.set = xg_param, control = rancontrol)
xg_final <- setHyperPars(learner = xg_learner, par.vals = xg_tune$x)
xgmodel <- mlr::train(xg_final, train_task_rose)
xgpredict <- predict(xgmodel, test_task)
nb_prediction <- xgpredict$data$response
dCM <- confusionMatrix(test_data$HasWriteOff, nb_prediction, positive = "2")
dCM 


# method 6 - SVM
getParamSet("classif.ksvm")
svm_learner <- makeLearner("classif.ksvm", predict.type = "response")
svm_param <- makeParamSet(
  makeDiscreteParam("C", values = 2^c(-8,-4,-2,0)), #cost parameters
  makeDiscreteParam("sigma", values = 2^c(-8,-4,0,4)) #RBF Kernel Parameter
)
ctrl <- makeTuneControlRandom()
cv_svm <- makeResampleDesc("CV",iters = 5L)
svm_tune <- tuneParams(svm_learner, task = train_task, resampling = cv_svm, par.set = svm_param, control = ctrl,measures = acc)
svm_tune$x
svm_tune$y
t.svm <- setHyperPars(svm_learner, par.vals = svm_tune$x)
svm_model <- mlr::train(svm_learner, train_task)
svmpredict <- predict(svm_model, test_task)
nb_prediction <- svmpredict$data$response
dCM <- confusionMatrix(test_data$HasWriteOff, nb_prediction, positive = "1")
dCM 

