# try MLE (maximum likelihood estimaiton) to predict hourly ticket selling count
# The goal for MLE is to get the coefficients of attributes used in function y
## Download the input data from https://s3-ap-south-1.amazonaws.com/av-blog-media/wp-content/uploads/2018/07/Train_Tickets.csv
library(data.table)

path<- "[YOUR INPUT FOLDER]"  # Use the folder path that stores the input file
setwd(path)

raw_data <- fread("MLE_tickets_sold.csv")
head(raw_data)

# The distribution can be treated as Poisson Distribution
hist(raw_data$Count, breaks = 50,probability = T ,main = "Histogram of Count Variable")
lines(density(raw_data$Count), col="red", lwd=2)

# Hourly sales count pattern
library(ggplot2)
library(lubridate)
ggplot(raw_data, aes(as.Date(dmy_hm(raw_data$Datetime)), Count)) + geom_line() +
  scale_x_date(date_labels = "%Y-%m-%d") + xlab("") + ylab("Daily Hourly Sales")

# Daily sales by adding up hourly sales in each day
raw_data <- raw_data[, extratced_data:=as.Date(dmy_hm(raw_data$Datetime))]
head(raw_data)
daily_count <- raw_data[,.(daily_sales=sum(Count)),by=extratced_data]
head(daily_count)
ggplot(daily_count, aes(extratced_data, daily_sales)) + geom_line() +
  scale_x_date(date_labels = "%Y-%m-%d") + xlab("") + ylab("Daily Sales")

# calculate elapsed_weeks (age)
raw_data[, elapsed_weeks:=as.double(difftime(raw_data$Datetime, min(raw_data$Datetime), units = "weeks"))]
head(raw_data)

# split into training & testing data with idx
library(caret)
set.seed(410)
idx <- createDataPartition(raw_data$Count, p=0.25, list = F)

# Method 1 - DIY to generate coefficients
library(stats4)  # Cannot be found in CRAN, no need to use installing tool
negative_likelihood <- function(theta0,theta1) {
  x <- raw_data$elapsed_weeks[-idx]
  y <- raw_data$Count[-idx]
  mu = exp(theta0 + x*theta1)
  return(-sum(y*(log(mu)) - mu))
}
maximumLikelihood_estimation <- mle(minuslogl = negative_likelihood, start = list(theta0=2, theta1=0))
summary(maximumLikelihood_estimation)
## with calculated coefficients (theta0, theta1), evaluate testing data
pred <- exp(coef(maximumLikelihood_estimation)['theta0'] + raw_data$Count[idx]*coef(maximumLikelihood_estimation)['theta1'])
mle_rmse <- RMSE(pred, raw_data$Count[idx])
mle_rmse  # 11.16034

# compare with other linear model
## log the Count to make it normally distributed
lm_fit <-  lm(log(Count)~elapsed_weeks, data=raw_data[-idx,])
summary(lm_fit)
pred_lm <- predict(lm_fit, raw_data[idx,])
lm_rmse <- RMSE(exp(pred_lm), raw_data$Count[idx])  # used log(Count) when building the model
lm_rmse  # 11.23529

# Method 2 - With glm, you can choose the distribution in "family", it will return coefficients directly
glm_fit <- glm(Count~elapsed_weeks, family = "poisson", data = raw_data[-idx,])
summary(glm_fit)
pred_glm <- predict(glm_fit,  raw_data[idx,])
glm_rmse <- RMSE(pred_glm, raw_data$Count[idx]) 
glm_rmse  # 14.72576
glm_rmse <- RMSE(exp(pred_glm), raw_data$Count[idx]) 
glm_rmse  # 10.69645
