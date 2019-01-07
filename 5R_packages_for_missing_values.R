# 5 R pacakges for dealing with missing values - I think R is an angel

# MICE package
## MICE assumes that the missing data are Missing at Random (MAR), 
## which means that the probability that a value is missing depends 
## only on observed value and can be predicted using them.
## use "?mice" to check different methods applied on different data types

attach(iris)
summary(iris)
# generate 10% missing values at random
library(missForest)
iris.mis <- prodNA(iris, noNA = 0.1)
summary(iris.mis)

library(mice)
md.pattern(iris.mis)   # show missing values in tabular form
library(VIM)   # show missing values in visualization
# the Missing Data graph shows the rank of features based on missing data numbers
mice_plot <- aggr(iris.mis, col=c('green', 'blue'),
                  numbers=TRUE, sortVars=TRUE,
                  labels=names(iris.mis), cex.axis=0.7,
                  gap=3, ylab=c("Missing Data", "Pattern"))

str(iris.mis)
# indicates which impute method to use for each column
imputed_data <- mice(iris.mis, method = c('pmm', 'pmm', 'pmm', 'pmm', 'polyreg'), 
                     seed=410)
summary(imputed_data)
# check imputed values
imputed_data$imp$Sepal.Width
# m in the summay of imputed_data indicates the number of data sets
# here, with complete(), we can choose a certain dataset
complete_data <- complete(imputed_data, 4)
summary(complete_data)
# build predictive model on all the m imputed datasets, use with() method
fit <- with(data = imputed_data, exp = lm(Sepal.Width ~ Sepal.Length+Sepal.Width))
# combine results of the analysis on m datasets
combine <- pool(fit)
summary(combine)



# Amelia package
## It assumpes that All variables in a data set have Multivariate Normal Distribution (MVN). 
## It uses means and covariances to summarize data. 
## It also assumes Missing data is random in nature (Missing at Random)
## This package works best when data has multivariable normal distribution
library(Amelia)
data("iris")
iris.mis <- prodNA(iris, noNA = 0.1)
summary(iris.mis)
amelia_imputed<- amelia(iris.mis, m = 5, parallel = "multicore", noms = "Species")

# check imputed outputs, or a specific column in an output
summary(amelia_imputed$imputations[[1]])
summary(amelia_imputed$imputations[[5]])
summary(amelia_imputed$imputations[[4]]$Sepal.Length)

# output the imputed data
write.amelia(amelia_imputed, file.stem = "amelia_imputed_data")



# missForest package
## It builds a random forest model for each variable. 
## Then it uses the model to predict missing values in the variable with the help of observed values.
library(missForest)
data("iris")
iris.mis <- prodNA(iris, noNA = 0.1)
summary(iris.mis)
# This one is so simple
missForest_imputed <- missForest(iris.mis)

# check imputed outputs
summary(missForest_imputed$ximp)
# check imputation error - NRMSE indicates the error for continuous values, PFC indicates categorical values
missForest_imputed$OOBerror
# compare actual imputation accuracy, in order to get lower error rate, 
## we could tune the params in missForest(), mtry and ntree
missForest_error <- mixError(missForest_imputed$ximp, iris.mis, iris)
missForest_error



# Hmisc package
## It assumes linearity in the variables being predicted.
library(Hmisc)
data("iris")
iris.mis <- prodNA(iris, noNA = 0.1)
summary(iris.mis)

# impute(), you can impute with mean, median, min, max, random values
## impute with mean
iris.mis$imputed_mean <- with(iris.mis, impute(Sepal.Length, mean))
summary(iris.mis$imputed_mean)
## impute with random values
iris.mis$imputed_random <- with(iris.mis, impute(Sepal.Length, 'random'))
summary(iris.mis$imputed_random)

# aregImpute(), detect the data type automatically
imputed_hmisc_areg <- aregImpute(~ Sepal.Length + Sepal.Width, data = iris.mis, n.impute = 5)
imputed_hmisc_areg
imputed_hmisc_areg_all <- aregImpute(~Sepal.Length + Sepal.Width + Petal.Length + Petal.Width + Species, 
                                     data = iris.mis, n.impute = 5)

imputed_hmisc_areg_all

# check imputed output
summary(imputed_hmisc_areg_all$imputed)
summary(imputed_hmisc_areg_all$imputed$Sepal.Length)



# mi package
## similar to pmm, for each observation in a variable with missing value, 
## we find observation (from available values)  with the closest predictive 
## mean to that variable. The observed value from this “match” 
## is then used as imputed value.
library(mi)
data("iris")
iris.mis <- prodNA(iris, noNA = 0.1)
summary(iris.mis)

mi_imputed <- mi(iris.mis, seed = 410)
summary(mi_imputed)


#----------------------------------------------#
# test when there is > 50% missing values
#----------------------------------------------#

data("iris")
library(missForest)
iris.mis <- prodNA(iris, noNA = 0.6)
summary(iris.mis)

mis1 <- iris.mis
mis2 <- iris.mis

# using missForest
missForest_imputed <- missForest(mis1, ntree = 100)
missForest_error <- mixError(missForest_imputed$ximp, mis1, iris)
dim(missForest_imputed$ximp)
missForest_error

# using Hmisc
library(Hmisc)
hmisc_imputed <- aregImpute(~Sepal.Length + Sepal.Width + Petal.Length + Petal.Width + Species, 
                            data = mis2, n.impute = 1)
length(hmisc_imputed$imputed$Sepal.Length)

check_missing <- function(x, hmisc) {
  # Check whether the index of origional missing data and the index of imputed data are the sames
  return(all.equal(which(is.na(x)), as.integer(attr(hmisc, "dimnames")[[1]])))
}

get_level_text <- function(val, lvls) {
  return(lvls[val])
}

convert <- function(miss_dat, hmisc) {
  m_p <- ncol(miss_dat)
  h_p <- length(hmisc)
  if (m_p != h_p) stop("miss_dat and hmisc must have the same number of variables")
  # assume matches for all if 1 matches
  if (!check_missing(miss_dat[[1]], hmisc[[1]]))
    stop("missing data and imputed data do not match")
  
  for (i in 1:m_p) {
    i_factor <- is.factor(miss_dat[[i]])
    if (!i_factor) {miss_dat[[i]][which(is.na(miss_dat[[i]]))] <- hmisc[[i]]}
    else {
      levels_i <- levels(miss_dat[[i]])
      miss_dat[[i]] <- as.character(miss_dat[[i]])
      miss_dat[[i]][which(is.na(miss_dat[[i]]))] <- sapply(hmisc[[i]], get_level_text, lvls= levels_i)
      miss_dat[[i]] <- factor(miss_dat[[i]])
    }
  }
  return(miss_dat)
}

mis2_converted <- convert(mis2, hmisc_imputed$imputed)
hmisc_error <- mixError(mis2_converted, mis2, iris)
hmisc_error
