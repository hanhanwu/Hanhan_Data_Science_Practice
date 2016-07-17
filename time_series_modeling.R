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

