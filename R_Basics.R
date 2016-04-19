#set working directory
path<- "/Users/hanhanwu/Desktop"
setwd(path)

# load data
train <- read.csv("R_basics_train.csv")
test <- read.csv("R_basics_test.csv")

# quick explore data
dim(train)
dim(test)
str(train)
str(test)

# check missisng data
table(is.na(train))
colSums(is.na(train))

# get more details of each feature
summary(train)

# Graphical Representation of Variables
library("ggplot2", lib.loc="/Library/Frameworks/R.framework/Versions/3.2/Resources/library")

ggplot(train, aes(x= Item_Visibility, y = Item_Outlet_Sales)) 
  + geom_point(size = 2.5, color="navy") + xlab("Item Visibility") + ylab("Item Outlet Sales")

ggplot(train, aes(Outlet_Identifier, Item_Outlet_Sales)) 
  + geom_bar(stat = "identity", color = "purple") 
  + theme(axis.text.x = element_text(size=6), axis.text.y = element_text(size=6), 
          panel.background = element_rect(fill = "white"), 
          panel.grid.major = element_line(colour = "grey"), 
          panel.grid.minor = element_blank())

ggplot(train, aes(Item_Type, Item_Outlet_Sales)) 
  + geom_bar( stat = "identity") 
  +theme(axis.text.x = element_text(angle = 70, vjust = 0.5), panel.grid.minor = element_blank())

ggplot(train, aes(Item_Type, Item_MRP)) 
  +geom_boxplot() +ggtitle("Box Plot") 
  + theme(axis.text.x = element_text(angle = 70, vjust = 0.5))

# combine training data and testing data
# An intuitive approach would be to extract 
# the mean value of targest from train data set and use it as placeholder for test, here just use 1
test$Item_Outlet_Sales <-  1
combi <- rbind(train, test)

# impute missing value using median, which is robust to outliers
# In this case, the evaluation method will be RMSE, which is highly affected by outliers, outliers is better
combi$Item_Weight[is.na(combi$Item_Weight)] <- median(combi$Item_Weight, na.rm = TRUE)
table(is.na(combi$Item_Weight))

# deal with abnormal values
##  1. Item_Visibility == 0 is impossible in practice 
combi$Item_Visibility <- ifelse(combi$Item_Visibility == 0,
                                median(combi$Item_Visibility), combi$Item_Visibility)

## 2. When there is missing categorical data, using "Other" to replace
levels(combi$Outlet_Size)
levels(combi$Outlet_Size)[1] <- "Other"
levels(combi$Outlet_Size)

## 3. When there are mismatched data, standaridize them
library("plyr", lib.loc="/Library/Frameworks/R.framework/Versions/3.2/Resources/library")
combi$Item_Fat_Content <- revalue(combi$Item_Fat_Content,
                                  c("LF" = "Low Fat", "reg" = "Regular"))
combi$Item_Fat_Content <- revalue(combi$Item_Fat_Content, c("low fat" = "Low Fat"))
table(combi$Item_Fat_Content)


# feature engineering
library("dplyr", lib.loc="/Library/Frameworks/R.framework/Versions/3.2/Resources/library")

## 1. creating count as a new column, with group by, merge is "inner join" here
a <- combi%>%
  group_by(Outlet_Identifier)%>%
  tally()
head(a)
names(a)[2] <- "Outlet_Count"
names(a)
combi <- merge(a, combi, by = "Outlet_Identifier")
dim(combi)

b <- combi%>%
  group_by(Item_Identifier)%>%
  tally()
names(b)[2] <- "Item_Count"
head (b)
combi <- merge(b, combi, by = "Item_Identifier")
dim(combi)

## 2. get time(year) difference 
c <- combi%>%
  select(Outlet_Establishment_Year)%>% 
  mutate(Outlet_Year = 2013 - combi$Outlet_Establishment_Year)
head(c)
# Here have to use generate d here, otherwise the merge will be slow and will create a huge df
d <- c%>%
  group_by(Outlet_Establishment_Year, Outlet_Year)%>%
  tally()
e <- select(d, Outlet_Establishment_Year, Outlet_Year)
combi <- merge(e, combi, by="Outlet_Establishment_Year")
dim(combi)

## 3. extract and rename the variables respectively
q <- substr(combi$Item_Identifier,1,2)
q <- gsub("FD","Food",q)
q <- gsub("DR","Drinks",q)
q <- gsub("NC","Non-Consumable",q)
table(q)

## 4. Label Encoding and One Hot Encoding
# a. Label Encoding - replace categorical data with numerical data
combi$Item_Fat_Content <- ifelse(combi$Item_Fat_Content == "Regular",1,0)

# b. One Hot Encoding - Split a column into unique columns based on unique values, 
# then remove the origional column
library("dummies", lib.loc="/Library/Frameworks/R.framework/Versions/3.2/Resources/library")
combi2 <- dummy.data.frame(combi, 
    names = c('Outlet_Size','Outlet_Location_Type','Outlet_Type', 'Item_Type_New'),  sep='_')
str (combi2)


# Predictive Modeling
# Linear Regression, higher R-square in summary, better the results
linear_model <- lm(Item_Outlet_Sales ~ ., data = new_train)
summary(linear_model)
# R-square is too small, check the problem. Note, cor() only works for numerical data
drops <- c("Item_Identifier","Item_Type", "Outlet_Identifier")
new_train_check <- new_train[ , !(names(new_train_check) %in% drops)]
cor(new_train_check)



# start over, without creating that much new columns
combi <- rbind(train, test)
combi$Item_Weight[is.na(combi$Item_Weight)] <- median(combi$Item_Weight, na.rm = TRUE)
combi$Item_Visibility <- ifelse(combi$Item_Visibility == 0,
                                median(combi$Item_Visibility), combi$Item_Visibility)
levels(combi$Outlet_Size)[1] <- "Other"
combi$Item_Fat_Content <- revalue(combi$Item_Fat_Content,
                                  c("LF" = "Low Fat", "reg" = "Regular"))
combi$Item_Fat_Content <- revalue(combi$Item_Fat_Content, c("low fat" = "Low Fat"))
combi$Year <- 2013 - combi$Outlet_Establishment_Year

#drop variables not required in modeling
combi <- select(combi, -c(Item_Identifier, Outlet_Identifier, Outlet_Establishment_Year))
new_train <- combi[1:nrow(train),]
new_test <- combi[-(1:nrow(train)),]

# Model 1: Linear Regression
linear_model <- lm(Item_Outlet_Sales ~ ., data = new_train)
summary(linear_model)
# improve model, check regression plot
par(mfrow=c(2,2))
plot(linear_model)
# try take the log of the targets
linear_model <- lm(log(Item_Outlet_Sales) ~ ., data = new_train)
summary(linear_model)
plot(linear_model)
# evaluate the model using RMSE
library("Metrics", lib.loc="/Library/Frameworks/R.framework/Versions/3.2/Resources/library")
rmse(new_train$Item_Outlet_Sales, exp(linear_model$fitted.values))


# Model 2: Decision Tree, R has to be >= 3.2.4
library("caret", lib.loc="/Library/Frameworks/R.framework/Versions/3.2/Resources/library")
library("rpart", lib.loc="/Library/Frameworks/R.framework/Versions/3.2/Resources/library")
library("e1071", lib.loc="/Library/Frameworks/R.framework/Versions/3.2/Resources/library")
library("rpart.plot", lib.loc="/Library/Frameworks/R.framework/Versions/3.2/Resources/library")

# set tree control params
fitControl <- trainControl(method = "cv", number = 5)
cartGrid <- expand.grid(.cp=(1:50)*0.01)

main_tree <- rpart(Item_Outlet_Sales ~ ., data = new_train, control = rpart.control(cp=0.01))
prp(main_tree)
pre_score <- predict(main_tree, type = "vector")
rmse(new_train$Item_Outlet_Sales, pre_score)


# Model 3: Random Forest
library("randomForest", lib.loc="/Library/Frameworks/R.framework/Versions/3.2/Resources/library")

## 1. parallel random forest, should be fast, but still very slow
control <- trainControl(method = "cv", number = 5)
rf_model <- train(Item_Outlet_Sales ~ ., data = new_train, method = "parRF", 
                  trControl = control, prox=TRUE,allowParallel=TRUE)
print(rf_model)

## 2. normal random forest
forest_model <- randomForest(Item_Outlet_Sales ~ ., data = new_train, mtry = 15, ntree = 1000)
print(forest_model)
varImpPlot(forest_model)
