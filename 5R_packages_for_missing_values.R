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
## It also auupmes Missing data is random in nature (Missing at Random)
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
