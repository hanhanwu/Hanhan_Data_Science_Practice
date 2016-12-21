# one-hot encoding for categorical data, then Boruta for feature selection

## reduce the levels of categorical data
ggplot(fact_data, aes(HasWriteOff, ..count..)) + geom_bar(aes(fill = TotalExtensions), position = "dodge")  
levels(fact_data$TotalExtensions)
levels(fact_data$TotalExtensions) <- c("0", rep("MoreThanZero", 12))

## One-Hot Encoding
library(dummies)
dummy_fact <- dummy.data.frame(fact_data)


## missing data has to be imputed before using Boruta
preProcValues <- preProcess(num_data, method = c("knnImpute","center","scale"))   # only deals with numerical data, normalized the data too
library('RANN')
num_data <- predict(preProcValues, num_data)

## Boruta feature selection
library(Boruta)
library('RANN')
q2 <- cbind(num_data, dummy_fact)
summarizeColumns(q2)
q2[, HasWriteOff0:= NULL]
q2[, HasWriteOff1:= NULL]

## remove 0 variance feature
zero_variance_list <- has_zero_variance(q2)
zero_variance_list

## remove highly correlation data
ax <-findCorrelation(x = cor(q2), cutoff = 0.7)   # 0.7 is the threshold here
summarizeColumns(q2)
sort(ax)
colnames(subset(q2, select = ax))    # get column names of those to be removed
q2 <- q2[, -ax, with=F]
q2$HasWriteOff <- as.factor(fact_data$HasWriteOff)
summarizeColumns(q2)

### boruta is using random forests for feature selection, by default
set.seed(1)
boruta_train <- Boruta(HasWriteOff~., data = q2, doTrace = 2)
boruta_train
### plot feature importance
plot(boruta_train, xlab = "", xaxt = "n")
str(boruta_train)
summary(boruta_train$ImpHistory)
finite_matrix <- lapply(1:ncol(boruta_train$ImpHistory), 
                        function(i) boruta_train$ImpHistory[is.finite(boruta_train$ImpHistory[,i]), i])
names(finite_matrix) <- colnames(boruta_train$ImpHistory)
plot_labels <- sort(sapply(finite_matrix, median))
axis(side = 1, las = 2, labels = names(plot_labels), 
     at = 1:ncol(boruta_train$ImpHistory), cex.axis = 0.7)
### determine tentative features
new_boruta_train <- TentativeRoughFix(boruta_train)
new_boruta_train
plot(new_boruta_train, xlab = "", xaxt = "n")
finite_matrix <- lapply(1:ncol(new_boruta_train$ImpHistory), 
                        function(i) new_boruta_train$ImpHistory[is.finite(new_boruta_train$ImpHistory[,i]), i])
names(finite_matrix) <- colnames(new_boruta_train$ImpHistory) 
plot_labels <- sort(sapply(finite_matrix, median))
axis(side = 1, las = 2, labels = names(plot_labels), 
     at = 1:ncol(new_boruta_train$ImpHistory), cex.axis = 0.7)
feature_stats = attStats(new_boruta_train)
feature_stats
selected_cols <- getSelectedAttributes(new_boruta_train, withTentative = F)
selected_cols <- selected_cols[1:41]  # remove HasWriteOff dummies
selected_cols


## model training with selected features
idx <- createDataPartition(q2$HasWriteOff, p=0.67, list=FALSE)
train_data <- q2[idx,]
test_data <- q2[-idx,]
selected_train <- subset(train_data, select = colnames(train_data) %in% selected_cols)
selected_train[, HasWriteOff:=train_data$HasWriteOff]
selected_test <- subset(test_data, select = colnames(test_data) %in% selected_cols)
selected_test[, HasWriteOff:=test_data$HasWriteOff]

train_task <- makeClassifTask(data=data.frame(selected_train), target = "HasWriteOff", positive = "1")
test_task <- makeClassifTask(data=data.frame(selected_test), target = "HasWriteOff", positive = "1")

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
dCM  

## check final feature importance
rf_feature_importance <- data.frame(rf_model$learner.model$importance)
feature_names <- data.frame(colnames(selected_train)[1:(length(selected_train)-1)])
colnames(feature_names)[1] <- 'Feature'
fi <- cbind(feature_names, rf_feature_importance$MeanDecreaseGini)
colnames(fi)[2] <- "MeanDecreaseGini"
setorder(fi, -"MeanDecreaseGini")
head(fi, n=15)
