library(astsa)
data()  # load all the data in package astsa

dim(jj)
nrow(jj)
ncol(jj)


jjm <- as.matrix(jj)
dim(jjm)
head(jj)
time(jj)
cycle(jj)
plot(jj, ylab="Earnings per Share", main="J & J")

# normal distribution for 99 random numbers
# start from 77th row, April
# display 12 months (if you set frequency as 4, it will play quarterly data)
(zardoz = ts(rnorm(99), start=c(77,4), frequency=12))   
(oz = window(zardoz, start=77, end=c(99,12)))   # use window to get part of the time series data
time(zardoz)
cycle(zardoz)
plot(zardoz, ylab="Random Value", main="Zardoz")

# too busy recently, TO BE CONTINUED...
