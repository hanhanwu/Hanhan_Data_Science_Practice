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


## TO BE CONTINUED...
