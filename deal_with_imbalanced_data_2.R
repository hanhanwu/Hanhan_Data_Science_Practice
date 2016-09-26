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


# In classification proble, it may help determine clusters when plotting the target
## with numeric variables
ggplot(data=num_train,aes(x = age, y=wage_per_hour))+geom_point(aes(colour=cat_train$income_level))+scale_y_continuous("wage per hour", breaks = seq(0,10000,1000))

## for the target with categorical variable, we could use bar chart
all_bar <- function(i){
  ggplot(cat_train,aes(x=i,fill=income_level))+geom_bar(position = "dodge",  color="black")+scale_fill_brewer(palette = "Pastel1")+theme(axis.text.x =element_text(angle  = 60,hjust = 1,size=10))
}

all_bar(cat_train$class_of_worker)
all_bar(cat_train$education)
# or use proportional table to check the target and the categorical variable
prop.table(table(cat_train$marital_status,cat_train$income_level),1)
prop.table(table(cat_train$class_of_worker,cat_train$income_level),1)


# data cleaning

## find missing data in numerical variables, no missing data in this case
table(is.na(num_train))
table(is.na(num_test))

## check correlation between numeric data
library(caret)
ax <-findCorrelation(x = cor(num_train), cutoff = 0.7)   # 0.7 is the threshold here
str(num_train)
num_train <- num_train[,-ax,with=FALSE] 
str(num_train)   # removed weeks_worked_in_year
num_test[,weeks_worked_in_year := NULL]
str(num_test)

## find missing data in categorical variables
mvtr <- sapply(cat_train, function(x){sum(is.na(x))/length(x)})*100
mvtr
mvte <- sapply(cat_test, function(x){sum(is.na(x))/length(x)})*100
mvte
#select columns with missing value less than 5%, 
## in this case, both training and testing data will remove the same columns
cat_train <- subset(cat_train, select = mvtr < 5 )
head(cat_train)
cat_test <- subset(cat_test, select = mvtr < 5 )


# TO BE CONTINUED....
