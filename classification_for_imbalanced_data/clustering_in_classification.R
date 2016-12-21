# clustering the data, and the cluster id becomes a new feature for the dataset

## selected features after Boruta feature selection
q3 <- subset(q2, select = colnames(q2) %in% selected_cols)

## clustering and the cluster id will be a new feature
set.seed(1)
kc <- kmeans(model.matrix(~.+0,data=q3), 6)
kc$cluster
q3[, Cluster:=kc$cluster]
q3[, HasWriteOff:=fact_data$HasWriteOff]
summarizeColumns(q3)
# There should be 1 cluster include as much HasWriteOff = 1 as possible
p1 <- q3$Cluster[which(q3$HasWriteOff==1)]
pt1 <- ggplot(data=data.frame(p1), aes(x= p1, y=..density..)) + geom_histogram(fill="blue",color="red",alpha = 0.5,bins =100) + geom_density()
p2 <- q3$Cluster[which(q3$HasWriteOff==0)]
pt2 <- ggplot(data=data.frame(p2), aes(x= p2, y=..density..)) + geom_histogram(fill="blue",color="red",alpha = 0.5,bins =100) + geom_density()
multiplot(pt1, pt2, cols=2)

## model training with selected features
idx <- createDataPartition(q3$HasWriteOff, p=0.67, list=FALSE)
train_data <- q3[idx,]
test_data <- q3[-idx,]

train_task <- makeClassifTask(data=data.frame(train_data), target = "HasWriteOff", positive = "1")
test_task <- makeClassifTask(data=data.frame(test_data), target = "HasWriteOff", positive = "1")

## Random Forest with boruta selected features
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
dCM <- confusionMatrix(test_data$HasWriteOff, nb_prediction, positive = "1")
dCM    # Balanced Accuracy:  0.95067, Sensitivity: 0.90698, Specificity: 0.99436

rf_feature_importance <- data.frame(rf_model$learner.model$importance)
feature_names <- data.frame(colnames(train_data)[1:(length(train_data)-1)])
colnames(feature_names)[1] <- 'Feature'
fi <- cbind(feature_names, rf_feature_importance$MeanDecreaseGini)
colnames(fi)[2] <- "MeanDecreaseGini"
setorder(fi, -"MeanDecreaseGini")
head(fi, n=15)
