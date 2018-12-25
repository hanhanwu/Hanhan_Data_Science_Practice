# try Prophet for time series prediction
install.packages('prophet')
install.packages('dplyr')  # update package
library(prophet)
library(data.table)
library(dplyr)
library(ggplot2)
library(lubridate)

path<- "[YOUR FOLDER PATH FOR INPUT DATA]"
setwd(path)

train <- fread("ts_PassengerTraffic_Train.csv")
test <- fread("ts_PassengerTraffic_Test.csv")
head(train)
head(test)
sapply(train, typeof)  # check data type
sapply(test, typeof)

# extract date from datetime string
train$Date <- as.POSIXct(strptime(train$Datetime, "%d-%m-%Y"))
test$Date <- as.POSIXct(strptime(test$Datetime, "%d-%m-%Y"))
head(train)
head(test)

# convert Datetime from character to datetime format
train$Datetime <- as.POSIXct(strptime(train$Datetime, "%d-%m-%Y %H:%M"))
test$Datetime <- as.POSIXct(strptime(test$Datetime, "%d-%m-%Y %H:%M"))
sapply(train, typeof)  # check data type
sapply(test, typeof)
head(train)

# Sum up daily counts
aggr_train <- train[,list(Count = sum(Count)), by = Date]
head(aggr_train)
ggplot(aggr_train) + geom_line(aes(Date, Count))  # plot daily count

# Change column names to what Prophet needs
names(aggr_train) = c("ds", "y")
head(aggr_train)

# Model prediction
m <- prophet(aggr_train, daily.seasonality=TRUE, seasonality.prior.scale=0.1)
periods <- dim(test)[1]/24
future <- make_future_dataframe(m, periods = periods)
forecast <- predict(m, future)
head(forecast)

plot(m, forecast)  # visualize the forecast

# calculate the fraction of hourly count
mean_hourly_count <- train %>%
group_by(hour = format(ymd_hms(train$Datetime), "%H")) %>%
summarise(mean_count = mean(Count))
head(mean_hourly_count)

total_avg_hourly_count <- sum(mean_hourly_count$mean_count)
mean_hourly_count$fraction <- mean_hourly_count$mean_count/total_avg_hourly_count
head(mean_hourly_count)

# convert daily forecast results to hourly in test data
sub_forecast <- forecast[, c('ds', 'yhat')]
head(sub_forecast)
test$hour <- format(ymd_hms(test$Datetime), "%H")
names(test) = c("ID", "Datetime", "ds", "hour")
head(test)

test_out <- merge(test, mean_hourly_count, by="hour")
test_out <- merge(sub_forecast, test_out, by="ds")
dim(test_out)
head(test_out)

test_out$prediction <- test_out$yhat * test_out$fraction
test_out <- test_out[, c("ID", "Datetime", "prediction")]
setorder(test_out, ID)
head(test_out)
