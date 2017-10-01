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

# Check data shifitng at the same time:
## https://github.com/hanhanwu/Hanhan_Data_Science_Practice/blob/master/deal_with_data_shifting.ipynb


# Method 1 - Filter Methods

  ## Check Feature Importance
  summary(q2$HasWriteOff)
  data_task <- makeClassifTask(data = data.frame(q2), target = "HasWriteOff", positive = "2")
  data_task <- removeConstantFeatures(data_task)
  ### gain ratio
  var_imp <- generateFilterValuesData(data_task, method = c("gain.ratio"))
  plotFilterValues(var_imp, feat.type.cols = TRUE)
  ### information gain
  var_imp <- generateFilterValuesData(data_task, method = c("information.gain"))
  plotFilterValues(var_imp, feat.type.cols = TRUE)
  ### anova.test
  var_imp <- generateFilterValuesData(data_task, method = c("anova.test"))
  plotFilterValues(var_imp, feat.type.cols = TRUE)

  ## create training, testing data based on the selected features
  selected_train <- subset(train_data, select = colnames(train_data) %in% selected_cols)
  selected_train[, HasWriteOff:=train_data$HasWriteOff]
  selected_test <- subset(test_data, select = colnames(test_data) %in% selected_cols)
  selected_test[, HasWriteOff:=test_data$HasWriteOff]



# Method 2 - Wrapper Methods, Caret Recursive Feature Selection

  control <- rfeControl(functions = rfFuncs,
                        method = "repeatedcv",
                        repeats = 3,
                        verbose = FALSE)
  outcomeName<-'HasWriteOff'
  predictors<-names(q2)[!names(q2) %in% outcomeName]
  recursive_selected_features <- rfe(data.frame(q2)[,predictors], data.frame(q2)[,outcomeName], rfeControl = control)
  recursive_selected_feature
  selected_cols <- c(4,8,16,61)
  selected_col_names <- colnames(q2_processed[, .SD, .SDcols=selected_cols])

  ## generate training, testing data based on selected features
  selected_train <- subset(train_data, select = colnames(train_data) %in% selected_col_names)
  selected_train[, HasWriteOff:=train_data$HasWriteOff]
  selected_test <- subset(test_data, select = colnames(test_data) %in% selected_col_names)
  selected_test[, HasWriteOff:=test_data$HasWriteOff]



# Method 3 - Boruta All-Relavant Feature Selection

  library(Boruta)
  set.seed(410)
  boruta_train <- Boruta(HasWriteOff~., data = train_data, doTrace = 2)
  boruta_train
  ## plot feature importance
  plot(boruta_train, xlab = "", xaxt = "n")
  str(boruta_train)
  summary(boruta_train$ImpHistory)
  finite_matrix <- lapply(1:ncol(boruta_train$ImpHistory), 
                          function(i) boruta_train$ImpHistory[is.finite(boruta_train$ImpHistory[,i]), i])
  names(finite_matrix) <- colnames(boruta_train$ImpHistory)
  plot_labels <- sort(sapply(finite_matrix, median))
  axis(side = 1, las = 2, labels = names(plot_labels), 
       at = 1:ncol(boruta_train$ImpHistory), cex.axis = 0.7)
  ## determine tentative features
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

  ## generate training, testing data based on selected features
  selected_train <- subset(train_data, select = colnames(train_data) %in% selected_cols)
  selected_train[, HasWriteOff:=train_data$HasWriteOff]
  selected_test <- subset(test_data, select = colnames(test_data) %in% selected_cols)
  selected_test[, HasWriteOff:=test_data$HasWriteOff]
                            
                            

# Method 4 - regression 
  glm_fit <- glm(formula = HasWriteOff ~ ., family = binomial(link = "logit"), data=q2_processed)
  sort(glm_fit$coefficients)
  glm_feature_coefficient <- data.frame(sort(glm_fit$coefficients, decreasing = T)) 
  colnames(glm_feature_coefficient) <- "coeffcient"
  feature_names <- data.frame(colnames(train_data)[1:(length(train_data)-1)])
  colnames(feature_names)[1] <- 'Feature'
  glm_fi <- cbind(feature_names, glm_feature_coefficient$coeffcient)
  colnames(glm_fi)[2] <- "coefficient"
  head(glm_fi, n=15)

  ## generate training, testing data based on selected features                           
  glm_selected_cols <- glm_fi$Feature[1:15]
  glm_selected_train <- subset(train_data, select = colnames(train_data) %in% glm_selected_cols)
  glm_selected_train[, HasWriteOff:=train_data$HasWriteOff]
  glm_selected_test <- subset(test_data, select = colnames(test_data) %in% glm_selected_cols)
  glm_selected_test[, HasWriteOff:=test_data$HasWriteOff]
