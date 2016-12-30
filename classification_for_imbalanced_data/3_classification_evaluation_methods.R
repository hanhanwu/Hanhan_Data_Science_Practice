library(verification)
library(mlr)

svm_learner <- makeLearner("classif.ksvm", predict.type = "prob")
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
svmpredict

# Method 1 - Balanced Accuracy
nb_prediction <- svmpredict$data$response
dCM <- confusionMatrix(test_data$HasWriteOff, nb_prediction, positive = "1")
dCM    # Balanced Accuracy:  0.95606, Sensitivity: 0.91489, Specificity: 0.99723

  ## Move Threshold
  svmpredict$threshold
  confusionMatrix(test_data$HasWriteOff, nb_prediction, positive = "1")

  pred2 <- setThreshold(svmpredict, 0.3)
  confusionMatrix(test_data$HasWriteOff, pred2$data$response, positive = "1")

  pred3 <- setThreshold(svmpredict, 0.4)
  confusionMatrix(test_data$HasWriteOff, pred3$data$response, positive = "1")

  pred4 <- setThreshold(svmpredict, 0.6)
  confusionMatrix(test_data$HasWriteOff, pred4$data$response, positive = "1")

  pred5 <- setThreshold(svmpredict, 0.7)
  confusionMatrix(test_data$HasWriteOff, pred5$data$response, positive = "1")

# Method 2 - Reliability Plot (this one may not be right)
xy <- data.table(Truth=svmpredict$data$truth, Response=svmpredict$data$response)
summary(xy$Truth)
summary(xy$Response)
xy[, ObservedFreq := ifelse(Truth==0, 1806/(1806+48), 48/(1806+48))]
xy[, ForecastedFreq := ifelse(Truth==0, 1807/(1807+47), 47/(1807+47))]
reliability.plot(svmpredict$data$prob.1, xy$ObservedFreq, xy$ForecastedFreq, positive="1")

# Method 3 - AUC
df <- generateThreshVsPerfData(svmpredict,measures = list(fpr,tpr))
plotROCCurves(df)

