library(SparkR)

# load data as Spark DataFrame
train <- read.df(sqlContext, "[R_basics_train.csv in your Spark HDFS path]",
                    source = "com.databricks.spark.csv", header="true", inferSchema = "true")
test <- read.df(sqlContext, "[R_basics_test.csv in your Spark HDFS path]",
                    source = "com.databricks.spark.csv", header="true", inferSchema = "true")
                    
# convert Spark DataFrame into R data.frame
train_df <- collect(train)
test_df <- collect(test)

# find missing data
colSums(is.na(train_df))

# quick explore data
dim(train_df)
dim(test_df)


# visualization using ggplot
library("ggplot2")
p1 <- ggplot(train_df, aes(Item_Visibility, Item_Outlet_Sales)) + geom_point(size = 2.5, color="navy") + xlab("Item Visibility") + ylab("Item Outlet Sales")
p1

p2 <- ggplot(train_df, aes(Outlet_Identifier, Item_Outlet_Sales)) + geom_bar(stat = "identity", color = "purple") + theme(axis.text.x = element_text(size=6), axis.text.y = element_text(size=6), panel.background = element_rect(fill = "white"), panel.grid.major = element_line(colour = "grey"), panel.grid.minor = element_blank())
p2

p3 <- ggplot(train_df, aes(Item_Type, Item_Outlet_Sales)) + geom_bar( stat = "identity") + theme(axis.text.x = element_text(angle = 70, vjust = 0.5), panel.grid.minor = element_blank())
p3

p4 <- ggplot(train_df, aes(Item_Type, Item_MRP)) + geom_boxplot() +ggtitle("Box Plot") + theme(axis.text.x = element_text(angle = 70, vjust = 0.5))
p4


# combine training data and testing data
# An intuitive approach would be to extract 
# the mean value of targest from train data set and use it as placeholder for test, here just use 1
test_df$Item_Outlet_Sales <- 1
dim(test_df)
dim(train_df)
combi <- rbind(train_df, test_df)
dim(combi)

# impute missing value using median, which is robust to outliers
# In this case, the evaluation method will be RMSE, which is highly affected by outliers, outliers is better
combi$Item_Weight[is.na(combi$Item_Weight)] <- median(combi$Item_Weight, na.rm = TRUE)
colSums(is.na(combi))

# deal with abnormal values
##  1. Item_Visibility == 0 is impossible in practice 
combi$Item_Visibility <- ifelse(combi$Item_Visibility == 0,
                                median(combi$Item_Visibility), combi$Item_Visibility)
max(combi$Item_Visibility)
min(combi$Item_Visibility)

## 2. When there is missing categorical data, using "Other" to replace
unique(combi$Outlet_Size)
combi$Outlet_Size <- ifelse(combi$Outlet_Size == "",
                                "Other", combi$Outlet_Size)
unique(combi$Outlet_Size)

## 3. When there are mismatched data, standaridize them
library("plyr")
unique(combi$Item_Fat_Content)
combi$Item_Fat_Content <- revalue(combi$Item_Fat_Content,
                                  c("LF" = "Low Fat", "reg" = "Regular"))
combi$Item_Fat_Content <- revalue(combi$Item_Fat_Content, c("low fat" = "Low Fat"))
unique(combi$Item_Fat_Content)

combi$Year <- 2013 - combi$Outlet_Establishment_Year

#drop variables not required in modeling
library("dplyr")
combi <- select(combi, -c(Item_Identifier, Outlet_Identifier, Outlet_Establishment_Year))

new_train <- combi[1:nrow(train_df),]
new_test <- combi[-(1:nrow(train_df)),]
dim(new_train)
dim(new_test)

# Model 1: Linear Regression
linear_model <- lm(Item_Outlet_Sales ~ ., data = new_train)
plot(linear_model) # can only show 1 plot even after using par(mfrow=c(2,2))
linear_model <- lm(log(Item_Outlet_Sales) ~ ., data = new_train)
plot(linear_model)

# does not support Metrics package, since Saprk R may have lower version in my community edition
# library("Metrics")
# rmse(new_train$Item_Outlet_Sales, exp(linear_model$fitted.values))

# does not support decision tree or rendom forest either since package "e1071" cannot be installed in the community edition
