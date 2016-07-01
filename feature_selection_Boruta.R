path<- "[you own file path]"
setwd(path)


library(Boruta)

train <- read.csv("loan_prediction_train.csv", header = TRUE, stringsAsFactors = FALSE)
str(train)
# replace one expression with another one
names(train) <- gsub("_", "", names(train))
str(train)
summary(train)


# Note! It's important to treat missing data BEFORE using Boruta package
train[train == ""] <- NA
colSums(is.na(train))
# indicate which cases are complete, having no missing data
train <- train[complete.cases(train),]
summary(train)
str(train)
# convert categorical data into factor - most simple to deal with missing data
convert_cols = c(2:6, 12:13)
train[,convert_cols] <- data.frame(apply(train[convert_cols], 2, as.factor))
summary(train)
colSums(is.na(train))

# Boruta is the most easy-to-use feature selection method I have used so far
# just 3 lines of code, it tells the important, unimportant features
set.seed(410)
boruta_train <- Boruta(LoanStatus~.-LoanID, data = train, doTrace = 2)
boruta_train

# plot the features in sorted median Z score order
## red, yellow, green indicate rejected, tentative and confirmed featues
## blue shows the min/max/mean Z score
plot(boruta_train, xlab = "", xaxt = "n")
str(boruta_train)
summary(boruta_train$ImpHistory)
finite_matrix <- lapply(1:ncol(boruta_train$ImpHistory), 
      function(i) boruta_train$ImpHistory[is.finite(boruta_train$ImpHistory[,i]), i])
names(finite_matrix) <- colnames(boruta_train$ImpHistory)
plot_labels <- sort(sapply(finite_matrix, median))
axis(side = 1, las = 2, labels = names(plot_labels), 
     at = 1:ncol(boruta_train$ImpHistory), cex.axis = 0.7)


# determine tentative features
## by comparing the median Z score of the tentative features with 
## the median Z score of the best shadow feature
new_boruta_train <- TentativeRoughFix(boruta_train)
new_boruta_train
plot(new_boruta_train, xlab = "", xaxt = "n")
finite_matrix <- lapply(1:ncol(new_boruta_train$ImpHistory), 
function(i) new_boruta_train$ImpHistory[is.finite(new_boruta_train$ImpHistory[,i]), i])
names(finite_matrix) <- colnames(new_boruta_train$ImpHistory) 
plot_labels <- sort(sapply(finite_matrix, median))
axis(side = 1, las = 2, labels = names(plot_labels), 
     at = 1:ncol(new_boruta_train$ImpHistory), cex.axis = 0.7)

getSelectedAttributes(new_boruta_train, withTentative = F)

feature_stats = attStats(new_boruta_train)
class(feature_stats)
feature_stats


# Traditional Feature Selection Algorithm
## RFE - Recursive Feature Elimination
library(caret)
library(randomForest)
set.seed(410)

# specific Random Forest as the underlying algorithm, just as Boruta
control <- rfeControl(functions = rfFuncs, method = "cv", number = 10)

rfe_train <- rfe(train[,2:12], train[,13], sizes = 1:12, rfeControl = control)
rfe_train
plot(rfe_train, type = c("g", "o"), cex = 1.0, col = 1:11)

predictors(rfe_train)
getSelectedAttributes(new_boruta_train, withTentative = F)
