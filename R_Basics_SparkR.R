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
