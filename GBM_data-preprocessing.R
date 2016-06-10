path<- "[your path of the data set]"
setwd(path)

library(data.table)

# using stringAsFactor so that categorical data will be easier to read in summary()
train <- fread("GBM_train.csv", stringsAsFactors = T)
test <- fread("GBM_test.csv", stringsAsFactors = T)

dim(train)
dim(test)

summary(train)
str(train)
colSums(is.na(train))

# get number of unique values, 698 different cities in this case
dim(unique(train[, .(City)]))
# remove City since there are too many values
train[, c("City") := NULL]
test[, c("City") := NULL]

# convert DOB to age
## convert to R standard Date format first
train$DOB <- as.character(train$DOB)
str(train$DOB)
ptn <- '(\\d\\d-\\w\\w\\w-)(\\d\\d)'
train$DOB <- sub(ptn, '\\119\\2', train$DOB)
str(train$DOB)
