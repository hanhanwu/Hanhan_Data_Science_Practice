library(data.table)
library(mlr)

path<- "[your folder]"
setwd(path)

data <- fread("loan_data.csv", na.strings = c("", " ", "?", "NA", NA))
summarizeColumns(data)

# seperate data into numericla and characterical
num_cols <- sapply(data, is.numeric)
num_data <- subset(data, select = num_cols==T)
fact_data <- subset(data, select = num_cols==F)

# impute the missing data in numerical data
summarizeColumns(num_data)
library(caret)
library('RANN')
set.seed(410)
imputed_num_cols <- preProcess(num_data, method = c("medianImpute","center","scale"))
imputed_num_data <- predict(imputed_num_cols, num_data)
summarizeColumns(imputed_num_data)

# impute the missing data in categorical data, as "MISSING"
summarizeColumns(fact_data)
for (i in seq_along(fact_data)) set(fact_data, i=which(is.na(fact_data[[i]])), j=i, value="MISSING")
summarizeColumns(fact_data)

imputed_data <- cbind(imputed_num_data, fact_data)
summarizeColumns(imputed_data)


# split into training and testing data, select specific cols
idx <- createDataPartition(imputed_data$Loan_Status, p=0.7, list=F)
trCols <- c("Credit_History", "LoanAmount", "Loan_Amount_Term", "ApplicantIncome", "CoapplicantIncome", "Loan_Status")
train_set <- imputed_data[idx, c(1:5, 13), with=F]
test_set <- imputed_data[-idx, c(1:5, 13), with=F]

train_task <- makeClassifTask(data=data.frame(train_set), target = "Loan_Status", positive = "Y")
test_task <- makeClassifTask(data=data.frame(test_set), target = "Loan_Status", positive = "Y")

# show all classifications can be used in mlr
listLearners("classif", check.packages = F)[c("class","package")]

# train multiple models, these are the bottom models in ensembling
## Random Forest
set.seed(410)
getParamSet("classif.randomForest")
rf_learner <- makeLearner("classif.randomForest", predict.type = "prob", par.vals = list(ntree = 200, mtry = 3))
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
dCM <- confusionMatrix(test_set$Loan_Status, nb_prediction, positive = "Y")
dCM 


## KNN
fitControl <- trainControl(
  method = "cv",
  number = 5,
  savePredictions = 'final',
  classProbs = T)
train_set$Loan_Status <- as.factor(train_set$Loan_Status)
test_set$Loan_Status <- as.factor(test_set$Loan_Status)

set.seed(410)
model_knn<-train(train_set[,c(1:5), with=F],train_set$Loan_Status,method='knn',trControl=fitControl,tuneLength=3)
knn_predict <- predict(object=model_knn, test_set[,c(1:5), with=F], type="prob")


## logistics regression
set.seed(410)
model_lr<-train(train_set[,c(1:5), with=F],train_set$Loan_Status,method='glm',trControl=fitControl,tuneLength=3)
lr_predict <- predict(object=model_lr, test_set[,c(1:5), with=F], type="prob")



# DIY Ensembling - Averaging Scores, have to use probability output
pred_avg <- (rfpredict$data$prob.Y + knn_predict$Y + lr_predict$Y)/3
## split into response output
test_set[, avg_pred := data.frame(as.factor(ifelse(pred_avg>0.5, 'Y', 'N')))]
head(test_set)

# DIY Ensembling - Majority Votes
test_set[, pred_rf:= rfpredict$data$response]
test_set[, pred_lr := predict(object=model_lr, test_set[,c(1:5), with=F])]
test_set[, pred_knn := predict(object=model_knn, test_set[,c(1:5), with=F])]
test_set[, pred_rf := rfpredict$data$response]
head(test_set)
test_set[, majority_pred:=data.frame(as.factor(ifelse(pred_rf=='Y' & pred_lr=='Y', 'Y', ifelse(pred_rf=='Y' & pred_knn=='Y', 'Y', ifelse(pred_knn=='Y' & pred_lr=='Y', 'Y', 'N')))))]
head(test_set)

# DIY Ensembling - Weighted Average, have to use probability output
pred_weighted_avg <- (rfpredict$data$prob.Y*0.5 + knn_predict$Y*0.25 + lr_predict$Y*0.25)
## split into response output
test_set[, weighted_avg_pred := data.frame(as.factor(ifelse(pred_weighted_avg>0.5, 'Y', 'N')))]
head(test_set)


# NOTE: if the prediction resultes of these bottom models are highly correlated, using them together may not bring better results
cor(data.frame(as.numeric(test_set$pred_knn), as.numeric(test_set$pred_lr), as.numeric(test_set$pred_rf)))



# DIY Stacking
## base models training
model_rf<-train(train_set[,c(1:5), with=F],train_set$Loan_Status,method='rf',trControl=fitControl,tuneLength=3)
model_lr<-train(train_set[,c(1:5), with=F],train_set$Loan_Status,method='glm',trControl=fitControl,tuneLength=3)
model_knn<-train(train_set[,c(1:5), with=F],train_set$Loan_Status,method='knn',trControl=fitControl,tuneLength=3)

##Predicting the out of fold prediction probabilities for training data
train_set$OOF_pred_rf<-model_rf$pred$Y[order(model_rf$pred$rowIndex)]
train_set$OOF_pred_knn<-model_knn$pred$Y[order(model_knn$pred$rowIndex)]
train_set$OOF_pred_lr<-model_lr$pred$Y[order(model_lr$pred$rowIndex)]

##Predicting probabilities for the test data
test_set$OOF_pred_rf<-predict(model_rf,test_set[,c(1:5), with=F],type='prob')$Y
test_set$OOF_pred_knn<-predict(model_knn,test_set[,c(1:5), with=F],type='prob')$Y
test_set$OOF_pred_lr<-predict(model_lr,test_set[,c(1:5), with=F],type='prob')$Y

## independent variables for the top layer
head(train_set)
predictors_top <- c(7:9)
model_gbm <- train(train_set[,c(1:5), with=F],train_set$Loan_Status,method='gbm',trControl=fitControl,tuneLength=3)

## predict using GBM top layer model
test_set$gbm_stacked<-predict(model_gbm,test_set[,predictors_top, with=F])
head(test_set)
