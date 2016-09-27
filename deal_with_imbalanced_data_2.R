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
tr(num_train$wage_per_hour)
tr(num_train$capital_gains)
tr(num_train$capital_losses)
tr(num_train$dividend_from_Stocks)
tr(num_train$num_person_Worked_employer)
tr(num_train$weeks_worked_in_year)


# In classification problem, it may help determine clusters when plotting the target
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
#select columns with missing value than 5%, 
## in this case, both training and testing data will remove the same columns
cat_train <- subset(cat_train, select = mvtr < 5 )
head(cat_train)
cat_test <- subset(cat_test, select = mvtr < 5 )
## to deal with the rest of missing data columns, convter NA to "Unavailable" (this solution, looks silly)
# convert to character
cat_train <- cat_train[,names(cat_train) := lapply(.SD, as.character),.SDcols = names(cat_train)]
for (i in seq_along(cat_train)) set(cat_train, i=which(is.na(cat_train[[i]])), j=i, value="Unavailable")
#convert back to factors
cat_train <- cat_train[, names(cat_train) := lapply(.SD,factor), .SDcols = names(cat_train)]
unique(cat_train$country_father)

# convert to character
cat_test <- cat_test[,names(cat_test) := lapply(.SD, as.character),.SDcols = names(cat_test)]
for (i in seq_along(cat_test)) set(cat_test, i=which(is.na(cat_test[[i]])), j=i, value="Unavailable")
#convert back to factors
cat_test <- cat_test[, names(cat_test) := lapply(.SD,factor), .SDcols = names(cat_test)]
unique(cat_test$country_father)


# deal with imbalanced data - combine levels with < 5% values in an imbalanced variable as "Other"
for(i in names(cat_train)){
  p <- 5/100
  ld <- names(which(prop.table(table(cat_train[[i]])) < p))
  levels(cat_train[[i]])[levels(cat_train[[i]]) %in% ld] <- "Other"
}

for(i in names(cat_test)){
  p <- 5/100
  ld <- names(which(prop.table(table(cat_test[[i]])) < p))
  levels(cat_test[[i]])[levels(cat_test[[i]]) %in% ld] <- "Other"
}


# check if categorical variables in training and testing data have mimatched levels
library(mlr)
summarizeColumns(cat_train)[,"nlevs"]
summarizeColumns(cat_test)[,"nlevs"]




# binning numerical variables to deal with data imbalance

# library(rpart)
# str(num_train)
# fit <- rpart(cat_train$income_level~ age+wage_per_hour+capital_gains+capital_losses+dividend_from_Stocks+num_person_Worked_employer, method="class", data=num_train)
# printcp(fit) # display the results 
# plotcp(fit) # visualize cross-validation results 
# summary(fit) # detailed summary of splits

num_train[,.N,age][order(age)]
tr(num_train$age)  # based on the plot, create 3 bins, 0-25, 26-50, 51-90
num_train[,age:= cut(x = age,breaks = c(0,25,50,90),include.lowest = TRUE,labels = c("young","adult","old"))]
num_train[,age := factor(age)]
unique(num_train$age)

num_test[,age:= cut(x = age,breaks = c(0,25,50,90),include.lowest = TRUE,labels = c("young","adult","old"))]
num_test[,age := factor(age)]
unique(num_test$age)


num_train[,.N,wage_per_hour][order(-N)]
  
  
# TO BE CONTINUED...
