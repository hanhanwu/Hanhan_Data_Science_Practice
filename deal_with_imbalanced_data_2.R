path<- "[your file path]"
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
train[,(factcols) := lapply(.SD, factor), .SDcols = factcols]
train[,(numcols) := lapply(.SD, as.numeric), .SDcols = numcols]

test[,(factcols) := lapply(.SD, factor), .SDcols = factcols]
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
tr(num_train$capital_gains)
tr(num_train$capital_losses)
tr(num_train$dividend_from_Stocks)
tr(num_train$num_person_Worked_employer)
tr(num_train$weeks_worked_in_year)


# TO BE CONTINUED....
