# 5 R pacakges for dealing with missing values - I think R is an angel

# MICE package
## MICE assumes that the missing data are Missing at Random (MAR), 
## which means that the probability that a value is missing depends 
## only on observed value and can be predicted using them.
## By default, linear regression is used to predict continuous missing values.
## Logistic regression is used for categorical missing values. 
## PMM (Predictive Mean Matching)  – For numeric variables
## logreg(Logistic Regression) – For Binary Variables( with 2 levels)
## polyreg(Bayesian polytomous regression) – For Factor Variables (>= 2 levels)
## Proportional odds model (ordered, >= 2 levels)
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


