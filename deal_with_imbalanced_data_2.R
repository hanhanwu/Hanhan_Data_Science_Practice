path<- "[YOUR FILE PATH]"
setwd(path)

library(data.table)

train <- fread("imbalance_train.csv", na.strings = c("", " ", "?", "NA", NA))
test <- fread("imbalance_test.csv", na.strings = c("", " ", "?", "NA", NA))

dim(train)
str(train)
# View(train)

dim(test)
str(test)

head(train)
head(test)

# check unique values of the target
unique(train$income_level)
unique(test$income_level)

# encode target values into 1 and 0, since it's binary here
train[,income_level := ifelse(income_level=="-50000", 0, 1)]
test[,income_level := ifelse(income_level=="-50000", 0, 1)]
unique(train$income_level)
unique(test$income_level)

# !!check severity of data imbalance, from taget values in the training data
round(prop.table(table(train$income_level))*100)

# convert multiple columns' data types, so convenient!
factcols <- c(2:5,7,8:16,20:29,31:38,40,41)
numcols <- setdiff(1:40,factcols)
train[,(factcols) := lapply(.SD, as.factor), .SDcols = factcols]
train[,(numcols) := lapply(.SD, as.numeric), .SDcols = numcols]

test[,(factcols) := lapply(.SD, as.factor), .SDcols = factcols]
test[,(numcols) := lapply(.SD, as.numeric), .SDcols = numcols]

str(train)
str(test)

cat_train <- train[,factcols, with=F]
cat_test <- test[,factcols, with=F]
num_train <- train[,numcols, with=F]
num_test <- test[,numcols, with=F]

rm(train, test)


library(ggplot2)
library(plotly)

# a plot function captures distribution pattern, with histogram and density curve
tr <- function(a){
  ggplot(data = num_train, aes(x= a, y=..density..)) + geom_histogram(fill="blue",color="red",alpha = 0.5,bins =100) + geom_density()
  ggplotly()
}

str(num_train)
tr(num_train$age)
tr(num_train$wage_per_hour)
tr(num_train$capital_gains)
tr(num_train$capital_losses)
tr(num_train$dividend_from_Stocks)
tr(num_train$num_person_Worked_employer)
tr(num_train$weeks_worked_in_year)


# In classification problem, it may help determine clusters when plotting the target
## with numeric variables
ggplot(data=num_train,aes(x = age, y=wage_per_hour))+geom_point(aes(colour=cat_train$income_level))+scale_y_continuous("wage per hour", breaks = seq(0,10000,1000))

## for the target with categorical variable, we could use bar chart
all_bar <- function(i){
  ggplot(cat_train,aes(x=i,fill=income_level))+geom_bar(position = "dodge",  color="black")+scale_fill_brewer(palette = "Pastel1")+theme(axis.text.x =element_text(angle  = 60,hjust = 1,size=10))
}

all_bar(cat_train$class_of_worker)
all_bar(cat_train$education)
# or use proportional table to check the target and the categorical variable
prop.table(table(cat_train$marital_status,cat_train$income_level),1)
prop.table(table(cat_train$class_of_worker,cat_train$income_level),1)


# data cleaning

## find missing data in numerical variables, no missing data in this case
table(is.na(num_train))
table(is.na(num_test))

## check correlation between numeric data
library(caret)
ax <-findCorrelation(x = cor(num_train), cutoff = 0.7)   # 0.7 is the threshold here
str(num_train)
num_train <- num_train[,-ax,with=FALSE] 
str(num_train)   # removed weeks_worked_in_year
num_test[,weeks_worked_in_year := NULL]
str(num_test)

## find missing data in categorical variables
mvtr <- sapply(cat_train, function(x){sum(is.na(x))/length(x)})*100
mvtr
mvte <- sapply(cat_test, function(x){sum(is.na(x))/length(x)})*100
mvte
#select columns with missing value than 5%, 
## in this case, both training and testing data will remove the same columns
cat_train <- subset(cat_train, select = mvtr < 5 )
head(cat_train)
cat_test <- subset(cat_test, select = mvtr < 5 )
## to deal with the rest of missing data columns, convter NA to "Unavailable" (this solution, looks silly)
# convert to character
cat_train <- cat_train[,names(cat_train) := lapply(.SD, as.character),.SDcols = names(cat_train)]
for (i in seq_along(cat_train)) set(cat_train, i=which(is.na(cat_train[[i]])), j=i, value="Unavailable")
#convert back to factors
cat_train <- cat_train[, names(cat_train) := lapply(.SD,factor), .SDcols = names(cat_train)]
unique(cat_train$country_father)

# convert to character
cat_test <- cat_test[,names(cat_test) := lapply(.SD, as.character),.SDcols = names(cat_test)]
for (i in seq_along(cat_test)) set(cat_test, i=which(is.na(cat_test[[i]])), j=i, value="Unavailable")
#convert back to factors
cat_test <- cat_test[, names(cat_test) := lapply(.SD,factor), .SDcols = names(cat_test)]
unique(cat_test$country_father)


# deal with imbalanced data - combine levels with < 5% values in an imbalanced variable as "Other"
for(i in names(cat_train)){
  p <- 5/100
  ld <- names(which(prop.table(table(cat_train[[i]])) < p))
  levels(cat_train[[i]])[levels(cat_train[[i]]) %in% ld] <- "Other"
}

for(i in names(cat_test)){
  p <- 5/100
  ld <- names(which(prop.table(table(cat_test[[i]])) < p))
  levels(cat_test[[i]])[levels(cat_test[[i]]) %in% ld] <- "Other"
}


# check if categorical variables in training and testing data have mimatched levels
library(mlr)
summarizeColumns(cat_train)[,"nlevs"]
summarizeColumns(cat_test)[,"nlevs"]




# binning numerical variables to deal with data imbalance

# library(rpart)
# str(num_train)
# fit <- rpart(cat_train$income_level~ age+wage_per_hour+capital_gains+capital_losses+dividend_from_Stocks+num_person_Worked_employer, method="class", data=num_train)
# printcp(fit) # display the results 
# plotcp(fit) # visualize cross-validation results 
# summary(fit) # detailed summary of splits

num_train[,.N,age][order(age)]
tr(num_train$age)  # based on the plot, create 3 bins, 0-25, 26-50, 51-90
num_train[,age:= cut(x = age,breaks = c(0,25,50,90),include.lowest = TRUE,labels = c("young","adult","old"))]
num_train[,age := factor(age)]
num_train[,.N,age][order(age)]

num_test[,age:= cut(x = age,breaks = c(0,25,50,90),include.lowest = TRUE,labels = c("young","adult","old"))]
num_test[,age := factor(age)]
num_test[,.N,age][order(age)]


## most values are 0, 2 bins (Zero, MoreThanZero)
num_train[,.N,wage_per_hour][order(-N)]
num_train[,wage_per_hour := ifelse(wage_per_hour == 0,"Zero","MoreThanZero")][,wage_per_hour := as.factor(wage_per_hour)]
num_train[,.N,wage_per_hour][order(-N)]

num_train[,.N,capital_losses][order(-N)]
num_train[,capital_losses := ifelse(capital_losses == 0,"Zero","MoreThanZero")][,capital_losses:= as.factor(capital_losses)]
num_train[,.N,capital_losses][order(-N)]

num_train[,.N,capital_gains][order(-N)]
num_train[,capital_gains := ifelse(capital_gains == 0,"Zero","MoreThanZero")][,capital_gains:= as.factor(capital_gains)]
num_train[,.N,capital_gains][order(-N)]

num_train[,.N,dividend_from_Stocks][order(-N)]
num_train[,dividend_from_Stocks := ifelse(dividend_from_Stocks == 0,"Zero","MoreThanZero")][,dividend_from_Stocks:= as.factor(dividend_from_Stocks)]
num_train[,.N,dividend_from_Stocks][order(-N)]


num_test[,.N,wage_per_hour][order(-N)]
num_test[,wage_per_hour := ifelse(wage_per_hour == 0,"Zero","MoreThanZero")][,wage_per_hour := as.factor(wage_per_hour)]
num_test[,.N,wage_per_hour][order(-N)]

num_test[,.N,capital_losses][order(-N)]
num_test[,capital_losses := ifelse(capital_losses == 0,"Zero","MoreThanZero")][,capital_losses:= as.factor(capital_losses)]
num_test[,.N,capital_losses][order(-N)]

num_test[,.N,capital_gains][order(-N)]
num_test[,capital_gains := ifelse(capital_gains == 0,"Zero","MoreThanZero")][,capital_gains:= as.factor(capital_gains)]
num_test[,.N,capital_gains][order(-N)]

num_test[,.N,dividend_from_Stocks][order(-N)]
num_test[,dividend_from_Stocks := ifelse(dividend_from_Stocks == 0,"Zero","MoreThanZero")][,dividend_from_Stocks:= as.factor(dividend_from_Stocks)]
num_test[,.N,dividend_from_Stocks][order(-N)]




# combine data
d_train <- cbind(num_train, cat_train)
d_test <- cbind(num_test, cat_test)
rm(num_train, num_test, cat_train, cat_test)

library(mlr) # The all in one library :)
# the task here is dataset
train.task <- makeClassifTask(data=d_train, target="income_level")
test.task <- makeClassifTask(data=d_test, target="income_level")

# remove zero variance features
train.task <- removeConstantFeatures(train.task)
test.task <- removeConstantFeatures(test.task)

# get variable importance chart
## This chart is deduced using a tree algorithm, where at every split, the information is calculated using reduction in entropy
var_imp <- generateFilterValuesData(train.task, method = c("information.gain"))
plotFilterValues(var_imp, feat.type.cols = TRUE)

# try undersampling, oversampling, SMOTE to balance data
## SMOTE: In SMOTE, the algorithm looks at n nearest neighbors, measures the distance between them and introduces a new observation at the center of n observations.
## undersampling: tends to loss of information
## oversampling: tends to overestimation of minority class

# undersampling
train_under <- undersample(train.task, rate = 0.1)   # keep 10% majority class
table(getTaskTargets(train_under))

# oversampling
train_over <- oversample(train.task, rate = 15)   # make minority class 15 times
table(getTaskTargets(train_over))

# SMOTE
train_smote <- smote(train.task, rate = 10, nn = 3)
table(getTaskTargets(train_smote))


# find available algorithms in MLR for the prediction problem here
listLearners("classif", "twoclass")[c("class", "package")]

## METHOD 1 - Try naive bayesian on all imbalanced, undersampled, oversmapled and SMOTE dataset,
## then compare accuracy using cross validation

naive_learner <- makeLearner("classif.naiveBayes", predict.type = "response")
naive_learner$par.vals <- list(laplace = 1)

## 10 folds for CV
folds <- makeResampleDesc("CV", iters=10, stratify = T)
fun_cv <- function(a){
  crv_val <- resample(naive_learner,a,folds,measures = list(acc,tpr,tnr,fpr,fp,fn))
  crv_val$aggr
}

## compare accuracy, tpr, tnr, fpr, fp, fn on the 4 dataset
fun_cv(train.task)
# acc.test.mean tpr.test.mean tnr.test.mean fpr.test.mean  fp.test.mean  fn.test.mean 
# 0.7267533     0.7153964     0.8984022     0.1015978   125.8000000  5326.1000000

fun_cv(train_under)
# acc.test.mean tpr.test.mean tnr.test.mean fpr.test.mean  fp.test.mean  fn.test.mean 
# 0.75964678    0.65667267    0.91527999    0.08472001  104.90000000  642.50000000 

fun_cv(train_over)
# acc.test.mean tpr.test.mean tnr.test.mean fpr.test.mean  fp.test.mean  fn.test.mean 
# 7.834291e-01  6.518080e-01  9.160502e-01  8.394982e-02  1.559200e+03  6.516100e+03 

fun_cv(train_smote)
# acc.test.mean tpr.test.mean tnr.test.mean fpr.test.mean  fp.test.mean  fn.test.mean 
# 8.709581e-01  8.205524e-01  9.471410e-01  5.285899e-02  6.545000e+02  3.358200e+03

# In this case, SMOTE gives the highest accuracy, use train_smote to build the model
nb_model <- train(naive_learner, train_smote)
nb_predict <- predict(nb_model, test.task)

## evaluate the model
nb_prediction <- nb_predict$data$response
dCM <- confusionMatrix(d_test$income_level, nb_prediction)
dCM
## F meansure
precision <- dCM$byClass['Pos Pred Value']
precision
recall <- dCM$byClass['Sensitivity']
recall
f_measure <- 2*((precision*recall)/(precision+recall))
f_measure

## According to dCM, Sensitivity is 98% means there is 98% accuracy in predicting majority class,
## but with 23% Specificity, which means minority class prediction accuracy is only 23%, 
## more models should be tried beyond Naive Bayesian


  
# XGBOOST
set.seed(410)
xgb_learner <- makeLearner("classif.xgboost", predict.type = "response")
xgb_learner$par.vals <- list(
  objective = "binary:logistic",
  eval_metric = "error",
  nrounds = 150,
  print.every.n = 50
)
## tuning params
xgb_params <- makeParamSet(
  makeIntegerParam("max_depth",lower=3,upper=10),
  makeNumericParam("lambda",lower=0.05,upper=0.5),
  makeNumericParam("eta", lower = 0.01, upper = 0.5),
  makeNumericParam("subsample", lower = 0.50, upper = 1),
  makeNumericParam("min_child_weight",lower=2,upper=10),
  makeNumericParam("colsample_bytree",lower = 0.50,upper = 0.80)
)
## random search function to choose params
rancontrol <- makeTuneControlRandom(maxit = 5L)  # 5 iterations
set_cv <- makeResampleDesc("CV", iters = 5L, stratify = T)  # 5 folds cross validation
## tune params
xgb_tune <- tuneParams(learner = xgb_learner, task = train.task, resampling = set_cv, measures = list(acc, tpr, tnr, fpr, fp, fn), par.set = xgb_params, control = rancontrol)
xgb_tune$x
## train the model and make predictions with optimal params
xgb_optimal <- setHyperPars(learner = xgb_learner, par.vals = xgb_tune$x)
xgb_model <- train(xgb_optimal, train.task)
xgb_predict <- predict(xgb_model, test.task)
xgb_prediction <- xgb_predict$data$response
## evaluate the prediction results
xgb_confusionmatrix <- confusionMatrix(d_test$income_level, xgb_prediction)
xgb_confusionmatrix    # Sensitivity: 0.9568, Specificity: 0.6657, Accuracy: 0.948
precision <- xgb_confusionmatrix$byClass["Pos Pred Value"]
precision
recall <- xgb_confusionmatrix$byClass["Sensitivity"]
recall
f_measure <- 2*((precision*recall)/(precision+recall))
f_measure     # 0.9726193 
## XGBoost got much better results than Naive Bayesian


## I'm trying to make XGBoost performs better, I'm going to use the top important features in the training data
## in the feature importance plot above, we have seen 20 top features have been chosen from 35 features
top_features <- filterFeatures(train.task, method = "information.gain", abs = 20)
xgb_model <- train(xgb_optimal, top_features)
xgb_predict <- predict(xgb_model, test.task)
xgb_prediction <- xgb_predict$data$response
## evaluate the prediction results
xgb_confusionmatrix <- confusionMatrix(d_test$income_level, xgb_prediction)
xgb_confusionmatrix    # Sensitivity: 0.938, Specificity: 0.0, Accuracy: 0.9379
precision <- xgb_confusionmatrix$byClass["Pos Pred Value"]
precision
recall <- xgb_confusionmatrix$byClass["Sensitivity"]
recall
f_measure <- 2*((precision*recall)/(precision+recall))
f_measure     # 0.9679402
## It seems that, after feature dimensional reduction, the performace dropped, 
## especially I got 0 Specificity here which means the minority group prediction is very bad


## Instead of predicint labels as above, I'm trying to predict probabilities below
xgb_prob <- setPredictType(learner = xgb_optimal, predict.type = "prob")
xgb_model_prob <- train(xgb_prob, train.task)
xgb_predict_prob <- predict(xgb_model_prob, test.task)
# check sample probability prediction results
xgb_predict_prob$data[1:10,]


# TUNE 1 - original threshold, 0.5
xgb_predict_prob$threshold
confusionMatrix(d_test$income_level, xgb_predict_prob$data$response)
## Sensitivity: 0.9569, Specifity: 0.6609

# TUNE 2 - threshold 0.4
pred2 <- setThreshold(xgb_predict_prob, 0.4)
confusionMatrix(d_test$income_level, pred2$data$response)
## Sensitivity: 0.9517, Specificity: 0.7148

# TUNE 3  - threshold 0.3
pred3 <- setThreshold(xgb_predict_prob, 0.3)
confusionMatrix(d_test$income_level, pred3$data$response)
## Sensitivity: 0.9466, Specificity: 0.7851

# TUNE 4  - threshold 0.6
pred4 <- setThreshold(xgb_predict_prob, 0.6)
confusionMatrix(d_test$income_level, pred4$data$response)
## Sensitivity: 0.9627, Specificity: 0.5977

# xgb threshold may influce the prediction for minority group when data is not balanced
## Using AUC plot to tune threshold, the AUC curve which is closer to the TOP LEFT CORNER is better
pt1 <- generateThreshVsPerfData(xgb_predict_prob, measures = list(fpr, tpr))
plotROCCurves(pt1)
pt2 <- generateThreshVsPerfData(pred3, measures = list(fpr, tpr))
plotROCCurves(pt2)
library(pROC)
roc <- plot(roc(pt1$data$tpr, pt1$data$fpr), print.auc = TRUE, col = "green")
roc <- plot(roc(pt2$data$tpr, pt2$data$fpr), print.auc = TRUE, col = "blue", print.auc.y = .4, add = TRUE)
## cannot tell too much differences in above plot.... use Sensitivity and Specifity numbers are better

## TUNE 3 is already good enough
# Beside tuning threshold, can try these methods too:
    # INcrease rounds
    # Use 10 folds CV
    # Increase repetitions in random search
    # Build xgb models on undersampling, oversampling, SMOTE data
    # Or, set weights to classes, set higher wieght to the class you want to pay more attention



# SVM
## check params
getParamSet("classif.svm")
svm_learner <- makeLearner("classif.svm", predict.type = "response")
## set weights to classes
svm_learner$par.vals <- list(class.weights = c("0"=1, "1"=10), kernel="radial")
svm_param <- makeParamSet(
  makeIntegerParam("cost", lower = 10^-1, upper = 10^2),
  makeIntegerParam("gamma", lower = 0.5, upper = 2)
)
## random search, cross validation settings
set_search <- makeTuneControlRandom(maxit = 5L)
set_cv <- makeResampleDesc("CV", iters=3L, stratify = T)
## find optimal params, looks like this step will take forever on my machine...
svm_optimal <- tuneParams(
  learner = svm_learner, 
  task = train.task, measures = list(acc, tpr, tnr, fpr, fp, fn),
  par.set = svm_param,
  control = set_search,
  resampling = set_cv
)
## train and predict
svm_model <- train(svm_optimal, train.task)
svm_predict <- predict(svm_model, test.task)
## evaluate
confusionMatrix(d_test$income_level, svm_predict$data$response)
