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
## one-hot encoding with City_Category
library(dummies)
combi <- dummy.data.frame(combi, names = c("City_Category"), sep = "_")

# check class/data types fo all variables
sapply(combi, class)
combi$Product_Category_2 <- as.integer(combi$Product_Category_2)
combi$Product_Category_3 <- as.integer(combi$Product_Category_3)

# data modeling with H2O
install.packages("h2o")
library(h2o)

new_train <- combi[1:nrow(train),]
new_test <- combi[1:nrow(test),]

# product_category_1 has 20 items while, product_category_2, 
# product_category_3 all have 18, so there maybe some noise in product_category_1
new_train <- new_train[new_train$Product_Category_1 <= 18,]


# H2O will use CPU memories
local_h2o <- h2o.init(nthreads = -1)
## check connection
h2o.init() 

## transfer data into H2O instance
h2o_train <- as.h2o(new_train)
h2o_test <- as.h2o(new_test)

# check column indexing number
colnames(h2o_train)
# Purchase (dependent variable/label) is #14
y.dep <- 14
# independent variables and drop IDs
x.indep <- c(3:13, 15:20)


# Multiple Regression in H2O
h2o_regression_model <- h2o.glm(y = y.dep, x = x.indep, training_frame = h2o_train, family = "gaussian")
h2o.performance(h2o_regression_model)
## low R2 here means that only 32.6% of the variance in the dependent variable
## is explained by independent variable and rest is unexplained. 
## This shows that regression model is unable to capture non linear relationships
reg_predict <- as.data.frame(h2o.predict(h2o_regression_model, h2o_test))
result <- data.frame(User_ID = test$User_ID, Product_ID = test$Product_ID, Purchase = reg_predict$predict)
# write.csv(result, "result.csv", row.names = F)


# Random Forest in H2O
h2o_random_forest_model <- h2o.randomForest(y = y.dep, x = x.indep, training_frame = h2o_train, 
                                            ntrees = 1000, mtries = 3,max_depth = 4,  seed = 1)
h2o.performance(h2o_random_forest_model)
## check variable importance
h2o.varimp(h2o_random_forest_model)
rf_predict <- as.data.frame(h2o.predict(h2o_random_forest_model, h2o_test))
result <- data.frame(User_ID = test$User_ID, Product_ID = test$Product_ID, Purchase = rf_predict$predict)
# write.csv(result, "result.csv", row.names = F)


# GBM in H2O, a boosting algorithm
h2o_gbm_model <- h2o.gbm(y = y.dep, x = x.indep, training_frame = h2o_train,
                         ntrees = 1000, max_depth = 4, learn_rate = 0.01, seed = 1)
h2o.performance(h2o_gbm_model)
h2o.varimp(h2o_gbm_model)
gbm_predict <- as.data.frame(h2o.predict(h2o_gbm_model, h2o_test))
result <- data.frame(User_ID = test$User_ID, Product_ID = test$Product_ID, Purchase = gbm_predict$predict)
# write.csv(result, "result.csv", row.names = F)


# Deep Learning in H2O
## 2 layers hidden layers here with 100 neurons each layer
## epochs is the number of passes on the train data to be carried out
## activation indicates the activation function
h2o_deep_learning_model <- h2o.deeplearning(y = y.dep, x = x.indep, training_frame = h2o_train,
                                            epochs = 60, hidden = c(100, 100), activation = "Rectifier", seed = 1)
h2o.performance(h2o_deep_learning_model)
dp_predict <- as.data.frame(h2o.predict(h2o_deep_learning_model, h2o_test))
result <- data.frame(User_ID = test$User_ID, Product_ID = test$Product_ID, Purchase = dp_predict$predict)
# write.csv(result, "result.csv", row.names = F)



# Using h2o.grid for param tuning, sort models through MSE, then get predicions for each model
grid <- h2o.grid("deeplearning", x = x.indep, y = y.dep, training_frame = h2o_train,
                 hyper_params = list(epochs = c(10, 60, 100), 
                                     hidden = c(100,100), 
                                     activation = c("Rectifier", "Maxout"),
                                     seed = 1))
grid   # get grid_id in the output
mse_table <- h2o.getGrid(grid_id = "Grid_DeepLearning_new_train_model_R_1463718746513_2",
                         sort_by = "mse", decreasing = TRUE)     
mse_table    # show details of model ranking and mses

model_ids <- grid@model_ids
models <- lapply(model_ids, function(id) { h2o.getModel(id) })
predictions <- lapply(models, function(m) { as.data.frame(h2o.predict(m, h2o_test)) })
