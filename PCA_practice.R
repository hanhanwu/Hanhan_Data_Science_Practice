#set working directory
path<- "[the root dir for your data sets]"
setwd(path)

# load data
train <- read.csv("PCA_train.csv")
test <- read.csv("PCA_test.csv")

# quick explore data
dim(train)
dim(test)
str(train)
str(test)
# check missisng data
table(is.na(train))
colSums(is.na(train))
summary(train)

#add a column
test$Item_Outlet_Sales <- 1

#combine the data set
combi <- rbind(train, test)

#impute missing values with median
combi$Item_Weight[is.na(combi$Item_Weight)] <- median(combi$Item_Weight, na.rm = TRUE)

# deal with abnormal values
##  1. Item_Visibility == 0 is impossible in practice 
combi$Item_Visibility <- ifelse(combi$Item_Visibility == 0,
                                median(combi$Item_Visibility), combi$Item_Visibility)

## 2. When there is missing categorical data, using "Other" to replace
levels(combi$Outlet_Size)
levels(combi$Outlet_Size)[1] <- "Other"
levels(combi$Outlet_Size)

levels(combi$Outlet_Identifier)
levels(combi$Item_Type)
levels(combi$Item_Identifier)

## 3. When there are mismatched data, standaridize them
library("plyr", lib.loc="/Library/Frameworks/R.framework/Versions/3.2/Resources/library")
combi$Item_Fat_Content <- revalue(combi$Item_Fat_Content,
                                  c("LF" = "Low Fat", "reg" = "Regular"))
combi$Item_Fat_Content <- revalue(combi$Item_Fat_Content, c("low fat" = "Low Fat"))
table(combi$Item_Fat_Content)

# PCA is unsupervised methods, remove label column and any identifier columns
my_data <- subset(combi, select = -c(Item_Outlet_Sales, Item_Identifier, Outlet_Identifier))

#check available variables
colnames(my_data)

# check data types
str(my_data)

# convert categorical data (Factor) into numerical data
# using one hot encoding
library(dummies)
#create a dummy data frame
new_my_data <- dummy.data.frame(my_data, names = c("Item_Fat_Content","Item_Type",
                                                   "Outlet_Establishment_Year","Outlet_Size",
                                                   "Outlet_Location_Type","Outlet_Type"))
str(new_my_data)

#principal component analysis (PCA)
## By default, it centers the variable to have mean equals to zero. 
## With parameter scale. = T, we normalize the variables to have standard deviation equals to 1.
prin_comp <- prcomp(new_my_data, scale. = T)
names(prin_comp)

# prcomp() function results in 5 useful measure
# 1. center and scale refers to respective mean and standard deviation 
## of the variables that are used for normalization prior to implementing PCA
prin_comp$center
prin_comp$scale

# 2. The rotation measure provides the principal component loading.
## Each column of rotation matrix contains the principal component loading vector.
prin_comp$rotation
# first 4 principal components and first 5 rows
prin_comp$rotation[1:5,1:4]

# 3. In order to compute the principal component score vector, we don’t need to multiply the loading with data. 
## Rather, the matrix x has the principal component score vectors in a 14204 × 44 dimension.
dim(prin_comp$x)

# plot the resultant principal components, 
## scale = 0 ensures that arrows are scaled to represent the loadings
biplot(prin_comp, scale = 0)

#compute standard deviation of each principal component
std_dev <- prin_comp$sdev

#compute variance
pr_var <- std_dev^2
#check variance of first 10 components
pr_var[1:10]

# higher is the explained variance, higher will be the information contained in those components
## proportion of variance explained
prop_varex <- pr_var/sum(pr_var)
prop_varex[1:20]

#scree plot
plot(prop_varex, xlab = "Principal Component",
       ylab = "Proportion of Variance Explained",
       type = "b")
# The plot above shows that ~ 30 components explains around 98.4% variance in the data set. 
## In order words, using PCA we have reduced 44 features to 30 without compromising on explained variance

# plot a cumulative variance plot, get give us a clear picture of number of components
plot(cumsum(prop_varex), xlab = "Principal Component",
     ylab = "Cumulative Proportion of Variance Explained",
     type = "b")
# This plot shows that 30 components results in variance close to ~ 98%.
## Therefore, in this case, we’ll select number of components as 30 [PC1 to PC30] 
## and proceed to the modeling stage. 
