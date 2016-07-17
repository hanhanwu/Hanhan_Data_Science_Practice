data("AirPassengers")
class(AirPassengers)

# This tells the format is time series
start(AirPassengers)
end(AirPassengers)
frequency(AirPassengers)
summary(AirPassengers)
str(AirPassengers)


# visualizing the data
plot(AirPassengers)
abline(reg = lm(AirPassengers~time(AirPassengers)))

## pritn cycle across years, then aggregate cycles and display a year on year trend
cycle(AirPassengers)
plot(aggregate(AirPassengers,FUN = mean))

## boxplot on seasonal data
boxplot(AirPassengers~cycle(AirPassengers))
## The variance and the mean value in July and August is much higher than rest of the months.
## Even though the mean value of each month is quite different their variance is small. 
## Hence, we have strong seasonal effect with a cycle of 12 months or less.


# stationization
## remove unequal variance and address trend component
library(tseries)
adf.test(diff(log(AirPassengers)), alternative = "stationary", k = 0)


# find parms for ARIMA model
## The ACF chart decays very slow, meaning the data is not stationary
acf(log(AirPassengers))
# Try diff
acf(diff(log(AirPassengers)))
pacf(diff(log(AirPassengers)))


# get param (p,d,q) values
## The value of p should be 0 as the ACF is the curve getting a cut off
## choose (p,d,q) that have both lowest AIC, BIC
p <- 0
d <- 1
min_sum <- 0
final_q <- 1
for (q in c(1,2)) {
  fit <- arima(log(AirPassengers), c(p,d,q), seasonal = list(order = c(p,d,q), period = 12))
  sum <- AIC(fit) + BIC(fit)
  if (sum < min_sum) {
    min_sum <- sum
    final_q <- q
  }
}

final_q

# make prediction, 2.718 is e
fit <- arima(log(AirPassengers), c(p,d,final_q), seasonal = list(order = c(p,d,final_q), period = 12))
pred <- predict(fit, n.ahead = 6*12)
ts.plot(AirPassengers, 2.718^pred$pred, log = "y", lty = c(1,3))
