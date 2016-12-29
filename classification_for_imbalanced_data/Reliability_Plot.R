############################################ Reliability Plot ##############################
# may not be right

library(verification)

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
xy <- data.table(Truth=svmpredict$data$truth, Response=svmpredict$data$response)
summary(xy$Truth)
summary(xy$Response)
xy[, ObservedFreq := ifelse(Truth==0, 1806/(1806+48), 48/(1806+48))]
xy[, ForecastedFreq := ifelse(Truth==0, 1807/(1807+47), 47/(1807+47))]
reliability.plot(svmpredict$data$prob.1, xy$ObservedFreq, xy$ForecastedFreq, positive="1")
