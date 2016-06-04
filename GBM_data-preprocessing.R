path<- "[your path of the data set]"
setwd(path)

library(data.table)

train <- fread("GBM_train.csv")
test <- fread("GBM_test.csv")

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


