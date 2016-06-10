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
## If simply use as.Date(), some years wil be convert to the wrong format, 
## I have to use regex here
train$DOB <- as.character(train$DOB)
str(train$DOB)
ptn <- '(\\d\\d-\\w\\w\\w-)(\\d\\d)'
train$DOB <- sub(ptn, '\\119\\2', train$DOB)
str(train$DOB)

test$DOB <- as.character(test$DOB)
test$DOB <- sub(ptn, '\\119\\2', test$DOB)
str(test$DOB)

## convert DOB to Age, default is Age in months
library(eeptools)
train$DOB <- as.Date(train$DOB, "%d-%b-%Y")
str(train$DOB)
train$DOB <- floor(age_calc(train$DOB, units = "years"))   # you may get warning, it's ok
str(train$DOB)
summary(train$DOB)

test$DOB <- as.Date(test$DOB, "%d-%b-%Y")
test$DOB <- floor(age_calc(test$DOB, units = "years")) 
summary(test$DOB)

## rename DOB as Age
train[, Age := DOB]
summary(train)
train[, DOB := NULL]
summary(train)

test[, Age := DOB]
test[, DOB := NULL]
summary(test)

# drop EmployerName, which has many distinct values
train[, Employer_Name := NULL]
summary(train)

test[, Employer_Name := NULL]
summary(train)
