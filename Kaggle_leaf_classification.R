path<- "[your file path]"
setwd(path)

library(mlr)
library(data.table)
train <- fread("train/train.csv", na.strings = c("", " ", "?", "NA", NA), stringsAsFactors = T)
test <- fread("test/test.csv", na.strings = c("", " ", "?", "NA", NA), stringsAsFactors = T)
sample <- fread("sample/sample_submission.csv")

summarizeColumns(train)
summarizeColumns(test)

test[, species := train$species[1:dim(test)[1]]]   # mock up label
train_ids <- train$id
test_ids <- test$id
train[, id := NULL]
test[, id := NULL]

## Method 1 - Bottom Line Prediction
train.task <- makeClassifTask(data=train, target="species")
test.task <- makeClassifTask(data=test, target="species")

set.seed(410)
xgb_learner <- makeLearner("classif.xgboost", predict.type = "prob")
xgb_learner$par.vals <- list(
  eval_metric = "merror",   # or mlogloss
  nrounds = 150,
  print.every.n = 50
)
xgb_prob <- setPredictType(learner = xgb_learner, predict.type = "prob")
xgb_model_prob <- train(xgb_prob, train.task)
xgb_predict_prob <- predict(xgb_model_prob, test.task)
# check sample probability prediction results
result <- xgb_predict_prob$data
