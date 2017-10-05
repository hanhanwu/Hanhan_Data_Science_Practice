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


# Method 0 - impute missing data with mode (for categorical data only)
sort(summary(dm_data$feature1)) .   # 'A' is the mode
dm_data$feature1[which(is.na(dm_data$feature1)==T)] = 'A'

# Method 1 - impute missing data with median/mode
## NOTE: This method does not impute missing data in the original data, but generate dummy columns
## and the values in dummy columns are normalized
q2 <- cbind(fact_data, num_data)
summarizeColumns(q2)
q2 <- mlr::impute(data.frame(q2), classes = list(factor = imputeMode(), integer = imputeMean()), dummy.classes = c("integer","factor"), dummy.type = "numeric")
q2 <- q2$data
summarizeColumns(q2)

# Method 2 - impute missing data in numerical data with median, caterogical data with a certain value
## NOTE: this method will normalize the data
library(caret)
library('RANN')
set.seed(410)
preProcValues <- preProcess(num_data, method = c("medianImpute","center","scale"))
imputed_data <- predict(preProcValues, num_data)

for (i in seq_along(fact_data)) set(fact_data, i=which(is.na(fact_data[[i]])), j=i, value="MISSING")

# Method 3 - impute missing data with KNN
## NOTE: this method will normalize the data
q2 <- cbind(fact_data, num_data)
summarizeColumns(q2)
fact_cols <- lapply(q2, is.factor)
fc <- colnames(subset(q2, select = fact_cols == T))
q2[,(fc) := lapply(.SD, as.numeric), .SDcols = fc]
q2$HasWriteOff <- as.factor(q2$HasWriteOff)
preProcValues <- preProcess(q2, method = c("knnImpute","center","scale"))   # only deals with numerical data, normalized the data too
library('RANN')
q2_processed <- predict(preProcValues, q2)
summarizeColumns(q2_processed)

# method 4 - DIY impute missing data
## If you don't want to change non-missing values, and you don't want to create any dummy columns,
## but just modify the missing values, try this method
impute_NA <- function(x) {
  if(is.numeric(x) == T) {
    x[is.na(x)] <- median(x[which(!is.na(x))])
  }
  else if(is.factor(x) == T) {
    x[is.na(x)] <- names(sort(summary(x), decreasing = T))[1]
  }
  return(x)
}

library(plyr)
my_data <- colwise(impute_NA)(my_data)  # here, don't use sapply() or other apply methods, they will convert all columns in to same data type
summarizeColumns(my_data)


# remove 0 variance feature
variance_lst <- nearZeroVar(q2, saveMetrics = T)
zero_variance_list <- names(subset(q2, select = variance_lst$zeroVar==T))
zero_variance_list
q2[, (zero_variance_list):=NULL]


# remove highly correlation data
fact_cols <- sapply(q2, is.factor)
q2_num_data <- data.table(subset(q2, select = fact_cols==F))
q2_fact_data <- data.table(subset(q2, select = fact_cols==T))
ax <-findCorrelation(x = cor(q2_num_data), cutoff = 0.7)   # 0.7 is the threshold here
summarizeColumns(q2_num_data)
sort(ax)
q2_num_data <- q2_num_data[, -ax, with=F]
q2 <- cbind(q2_num_data, q2_fact_data)
rm(q2_num_data)
rm(q2_fact_data)

  
# Normalize data into [0,1] scale
## KNN will normlize data while imputing missing data, however, it is not [0, 1] scale
data_scaling <- function(x){(x-min(x))/(max(x)-min(x))}
scaled_data <- data.frame(sapply(q2_num_data, data_scaling))
summarizeColumns(scaled_data)

## an alternative way to normalize to [0,1] range using min-max normalization (this works pretty well)
max = apply(cereal_data, 2, max)
max
min = apply(cereal_data, 2, min)
min
scaled_data <- as.data.table(scale(cereal_data, center = min, scale = max-min))
  
# Normalize data into like KNN, witout using KNN
scaled_data <- data.table(scale(q2_num_data))

  
# Method 1 - deal with outliers with median/mode
boxplot(q2$LoanAmount)
q2$LoanAmount[which(q2$LoanAmount>12)] <- median(q2$LoanAmount)
boxplot(q2$LoanAmount)

# Method 2 - deal with outliers with binning
boxplot(q2$mlbf_InterestRate)
num_distribution_plot(q2$mlbf_InterestRate, q2)
num_distribution_plot(sqrt(q2$mlbf_InterestRate), q2)
quantile(q2$mlbf_InterestRate)
q2$mlbf_InterestRate <- as.factor(ifelse (q2$mlbf_InterestRate <= 0.05, "0~0.05", ifelse(q2$mlbf_InterestRate >0.07, ">0.07", "0.05~0.07")))
summary(q2$mlbf_InterestRate)

# Better binning methods, check feature variance for bin
## Numerical Data
## NOTE: group_by here comes from library(dplyr), if you loaded library(plyr) after dplyr, group_by won't work
### so, use detach(package:plyr), then it should work
num_distribution_plot(dm_data$num_feature, dm_data)
quantile(dm_data$num_feature)
bin_num <- 7
dm_data[, num_feature_bin := as.integer(cut2(dm_data$num_feature, g = bin_num))] # binning, generates a new col as for bin_ids
summary(as.factor(dm_data$num_feature_bin))
temp_variance <- data.table(group_by(dm_data, color_group) %>%  # change color_group to your group by col
                              summarise(GroupVariance=var(rep(num_feature_bin)))) # feature variance for group
temp_variance
var(dm_data$num_feature_bin)   # feature variance for the whole
summary(cut2(dm_data$num_feature, g = bin_num))

### group by multiple cols
temp_variance <- data.table(group_by(dm_data, color_group, shape_group) %>%  # change color_group, shape_group to your group by cols
                              summarise(GroupVariance=var(rep(num_feature_bin)))) # feature variance for group

## Categorical Data
dm_data[, cat_feature_int:= as.integer(dm_data$cat_feature)] # generate an int col first
num_distribution_plot(dm_data$cat_feature, dm_data)
quantile(dm_data$cat_feature_int)
bin_num <- 9
dm_data[, bin_cat_feature_int := as.integer(cut2(dm_data$cat_feature_int, g = bin_num))]
temp_variance <- data.table(group_by(dm_data, color_group) %>%  # change color_group to your group by col
                              summarise(GroupVariance=var(rep(bin_cat_feature_int))))  # feature variance for group
temp_variance
var(dm_data$bin_cat_feature_int) 
summary(cut2(dm_data$cat_feature_int, g = bin_num))
sort(summary(dm_data$cat_feature))
sort(summary(as.factor(dm_data$cat_feature_int)))

# Method 3 - deal with outliers in a more statistical method (IQR)
# impute outliers with median
impute_outlier_median <- function(x) {
  x[x < quantile(x,0.25) - 1.5 * IQR(x) | x > quantile(x,0.75) + 1.5 * IQR(x)] <- median(x)
  return(x)
}

# impute outliers with mix, max
impute_outlier_min_max <- function(x) {
  x[x < quantile(x,0.25) - 1.5 * IQR(x)] <- min(quantile(x,0.25) - 1.5 * IQR(x))
  x[x > quantile(x,0.75) + 1.5 * IQR(x)] <- max(quantile(x,0.75) + 1.5 * IQR(x))
  return(x)
}
boxplot(dm_data$feature1)
quantile(dm_data$feature1)  # check quantile before using log(), in case to create infinity
# dm_data$feature1 <- impute_outlier_median(dm_data$feature1)  # impute with median
dm_data$feature1 <- impute_outlier_min_max(dm_data$feature1)  # impute with min, max
boxplot(dm_data$feature1)
sd(dm_data$feature1)   # standard deviation, the squared root of variance
var(dm_data$feature1)  # variance
num_distribution_plot(dm_data$feature1, dm_data) # the function created in data_explore


# Method 1 - remove almost constant varibales
get_rare_case <- function(a, n) {
  if (is.numeric(a)){
    return(length(a[which(a!=0)]) <= n)
  }
  else if (is.factor(a) & length(levels(q2$IsDelinquent_v1))==2){
     return(length(a[which(a==levels(a)[1])]) <= n | length(a[which(a==levels(a)[2])]) <= n)
  }
  return(FALSE)
}

rare_cases <- subset(q2, select = sapply(q2, get_rare_case, 3) ==T)
colnames(rare_cases)
q2[, names(rare_cases):=NULL]
  
# Method 2 - remove near 0 variance variables, used nearZeroVar() function
near_zero_variance_list <- c(2, 4, 7, 9, 10, 13, 14, 15, 22, 26)
smote_train_data <- scaled_train_data[, -near_zero_variance_list, with=F]
smote_test_data <- scaled_test_data[, -near_zero_variance_list, with=F]
