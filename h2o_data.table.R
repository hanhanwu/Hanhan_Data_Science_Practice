path<- "[your data file path]"
setwd(path)

library(data.table)

# Purchase is the label column
train <- fread("train.csv", stringsAsFactors = T)
test <- fread("test.csv", stringsAsFactors = T)

# combine dataset, using rbinflist() is faster than rbind
test[,Purchase := mean(train$Purchase)]    # add Purchase column in test first, so that it shares the same number of columns with train
c <- list(train, test)
combi <- rbindlist(c)

# data exploration with data.table
## category distribution
combi[,prop.table(table(combi$Gender))]
combi[,prop.table(table(combi$Age))]
combi[,prop.table(table(combi$City_Category))]
combi[,prop.table(table(combi$Stay_In_Current_City_Years))]
## unique values
length(unique(combi$User_ID))
length(unique(combi$Product_ID))
## missing values
colSums(is.na(combi))


library(ggplot2)
# Age and Gender
ggplot(combi, aes(Age, fill = Gender)) + geom_bar()
# Age and City_Category
ggplot(combi, aes(Age, fill = City_Category)) + geom_bar()
# cross tables for categorical variables
library(gmodels)
CrossTable(combi$Occupation, combi$City_Category)


# Product_Category2 and Product_Category3 have many NAs, may have hidden trend, 
# create new columns for the missing values
combi[, Product_Category_2_NA := ifelse(sapply(combi$Product_Category_2, is.na) == TRUE, 1,0)]
combi[, Product_Category_3_NA := ifelse(sapply(combi$Product_Category_3, is.na) == TRUE, 1,0)]
# impute missing values in the original columns
combi[, Product_Category_2 := ifelse(is.na(combi$Product_Category_2) == TRUE, "-999", combi$Product_Category_2)]
combi[, Product_Category_3 := ifelse(is.na(combi$Product_Category_3) == TRUE, "-999", combi$Product_Category_3)]

# revalue variables
levels(combi$Stay_In_Current_City_Years)[levels(combi$Stay_In_Current_City_Years) == "4+"] <- 4
levels(combi$Age)[levels(combi$Age) == "0-17"] <- 0
levels(combi$Age)[levels(combi$Age) == "18-25"] <- 1
levels(combi$Age)[levels(combi$Age) == "26-35"] <- 2
levels(combi$Age)[levels(combi$Age) == "36-45"] <- 3
levels(combi$Age)[levels(combi$Age) == "46-50"] <- 4
levels(combi$Age)[levels(combi$Age) == "51-55"] <- 5
levels(combi$Age)[levels(combi$Age) == "55+"] <- 6

combi$Age <- as.numeric(combi$Age)
combi[,Gender := as.numeric(combi$Gender) -1]

# feature engineering

## User_Count: 1+ means a user purchased multiple times
combi[, User_Count := .N, by = User_ID]
## Product_Count: 1+ means a product has been purchased multiple times
combi[, Product_Count := .N, by = Product_ID]
## Mean_Purchase_Product
combi[, Mean_Purchase_Product := mean(Purchase), by = Product_ID]
## Mean_Purchase_User
combi[, Mean_Purchase_User := mean(Purchase), by = Product_ID]
## one hot encoding with City_Category
library(dummies)
combi <- dummy.data.frame(combi, names = c("City_Category"), sep = "_")

# check class/data types fo all variables
sapply(combi, class)
combi$Product_Category_2 <- as.integer(combi$Product_Category_2)
combi$Product_Category_3 <- as.integer(combi$Product_Category_3)

# data modeling with H2O
install.packages("h2o")
library(h2o)




