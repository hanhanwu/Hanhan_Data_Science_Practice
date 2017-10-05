# Try neural network in R
## about the data: http://lib.stat.cmu.edu/DASL/Datafiles/Cereals.html
## download the data: https://s3-ap-south-1.amazonaws.com/av-blog-media/wp-content/uploads/2017/09/07122416/cereals.csv
library(mlr)
library(data.table)
library(caret)
library(neuralnet)
library(boot)
library(plyr)
library(matrixStats)

path<- "[change to your own path]"
setwd(path)

cereal_data <- fread("cereals.csv", na.strings = c("", " ", "?", "NA", NA))
summarizeColumns(cereal_data)

# split into training and testing data
idx <- createDataPartition(cereal_data$rating, p=0.7, list=FALSE)
train_data <- cereal_data[idx,]
test_data <- cereal_data[-idx,]

# normalize/scale the data (min-max normalizaton, to [0,1] range)
max = apply(cereal_data, 2, max)
max
min = apply(cereal_data, 2, min)
min
scaled_data <- as.data.table(scale(cereal_data, center = min, scale = max-min))
summarizeColumns(scaled_data)

# train NN
trainNN <- scaled_data[idx,]
testNN <- scaled_data[-idx,]

set.seed(410)
NN_model <- neuralnet(rating ~ calories + protein + fat + sodium + fiber, 
                      trainNN, hidden = 3, linear.output = T)
plot(NN_model)  # Error: 0.097287, Steps: 121

# predict
predict_NN <- compute(NN_model, testNN[,c(1:5)])
predict_NN
## reverse to original scale
reversed_scale_predict_NN <- predict_NN$net.result*(max(cereal_data$rating) - min(cereal_data$rating)) + min(cereal_data$rating)
## plot prediction and ground truth
plot(test_data$rating, reversed_scale_predict_NN, col='pink', pch=16, 
     ylab="Prediction Result", xlab = "Ground Truth")
abline(0,1)

# Root Mean Square Error (RMSE)
RMSE_NN <- sqrt(sum((test_data$rating - reversed_scale_predict_NN)^2)/nrow(test_data))
RMSE_NN  # 5.247114573

# DIY k-fold cross validation for NN
set.seed(410)
k=100
RMSE_NN <- NULL
RMSE_lst <- list()

dim(cereal_data) # 75, 6

for(j in 10:65){ # using random sample to choose index, ave 10 left at head and tail
  for(i in 1:k){  # each fold had 100 random run
    idx <- sample(1:nrow(cereal_data),j)
    trainNN <- scaled_data[idx,]
    testNN <- scaled_data[-idx,]
    test_data <- cereal_data[-idx,]
    
    NN_model <- neuralnet(rating ~ calories + protein + fat + sodium + fiber, 
                          trainNN, hidden = 3, linear.output = T)
    predict_NN <- compute(NN_model, testNN[,c(1:5)])
    reversed_scale_predict_NN <- predict_NN$net.result*(max(cereal_data$rating) - min(cereal_data$rating)) + min(cereal_data$rating)
    
    RMSE_NN[i] <- sqrt(sum((test_data$rating - reversed_scale_predict_NN)^2)/nrow(test_data))
  }
  RMSE_lst[[j]] <- RMSE_NN
}

RMSE_matrix <- do.call(cbind, RMSE_lst)
RMSE_matrix
# boxplot a certain column in RMSE matrix
# It means, when training data has length of 60, RMSE is 5.8
boxplot(RMSE_matrix[,51], ylab = "RMSE", main = "RMSE BoxPlot (length of traning set = 60)")

med <- colMedians(RMSE_matrix)
X = seq(10,65)
plot(med~X, type = "l", xlab = "Length of Training Set", ylab = "Median of RMSE",
     main = "RMSE changes along with the Length of Training data")

# Throught the plot above we found when training data length increase, RMSE decreases a lot
dim(RMSE_matrix)
RMSE_matrix[nrow(RMSE_matrix), ncol(RMSE_matrix), drop = FALSE]  # 3.84215857, do lower than the RMSE without cross validation above
