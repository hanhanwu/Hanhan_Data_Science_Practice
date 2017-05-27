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
plot(jj, type="o", col="pink", lty="dashed")
plot(diff(log(jj)), main="logged and diffed") 

# normal distribution for 99 random numbers
# start from 77th row, April
# display 12 months (if you set frequency as 4, it will play quarterly data)
(zardoz = ts(rnorm(99), start=c(77,4), frequency=12))   
(oz = window(zardoz, start=77, end=c(99,12)))   # use window to get part of the time series data
time(zardoz)
cycle(zardoz)
plot(zardoz, ylab="Random Value", main="Zardoz")


# plot(), plot.ts() and ts.plot()
x = -5:5
y = 5*cos(x) 
x
y
plot(x, main="plot(x)")
plot(x, y, main="plot(x,y)")

plot.ts(x, main="plot.ts(x)")
plot.ts(x, y, main="plot.ts(x,y)")

ts.plot(x, main="ts.plot(x)")
ts.plot(ts(x), ts(y), main="ts.plot(x,y)")   # it plots x, y seperately, along the time

k = c(.5,1,1,1,.5)        
(k = k/sum(k))       
# Applies linear filtering to the univariate time serie
# If sides = 1 the filter coefficients are for past values only; if sides = 2 they are centred around lag 0, 
fjj = filter(jj, sides=2, k)
dljj = diff(log(jj))   # difference the logged data
plot(jj)
lines(fjj, col="red")
lines(lowess(jj), col="blue", lty="dashed")
lines(dljj, col="green")

# check normality, I think histogram is always a good way to show normality
shapiro.test(dljj) 

par(mar = rep(2, 4))        # set up the graphics frame
hist(dljj, prob=TRUE, 12)   # histogram    
lines(density(dljj))     # smooth it
qqnorm(dljj)             # normal Q-Q plot  
qqline(dljj)             # add a line 


# lag.plot is used to show the correlation between the series data and the lag values of the series
## I think this method is not as straightforward as ACF, PACF visualization, but just want to try it
lag1.plot(dljj, 4)
lag1.plot(dljj, 5)
lag1.plot(dljj, 7)
lag1.plot(dljj, 9)


# This function gives both ACF and PACF visualization
acf2(dljj)   # AR model
acf2(log(jj))  # neither AR nor MA


# this visualization is a structural decomposition of season, trend and error (lag)
plot(cat <- stl(dljj, "per"))  # this means periodically seasonal extraction
summary(cat)
plot(cat <- stl(log(jj), "per"))


# Method 1 to fit the regression
Q <- factor(cycle(jj))   # quarter factors (1,2,3,4)
Q
trend <- time(jj) - 1970  # almost center the time, to make the output nicer
trend
reg <- lm(log(jj)~ 0 +trend + Q, na.action = NULL)
summary(reg)   # NOTE: the number of * equals to 1/p-value
options(show.signif.stars = F)   # you can also turn off the stars
summary(reg)
model.matrix(reg)

plot(log(jj), type="o")
lines(fitted(reg), col=2)  # looks almost fitted, let's check residuals now

par(mfrow = c(2,1))
plot(resid(reg))
acf2(resid(reg), 20)   # the residual doesn't look white (random), so it's not really fit
# to be honest, log(jj) is not stationary, we should not use it directly.


# Method 2 to fit the model
# Here, I'm trying sarima to fit the model. Because it should be an AR model, so I set q as 0
## Now the code is tryign to get p at the lowest sum(AIC, BIC)
q <- 0
d <- 1
min_sum <- 0
final_p <- 1
for (p in c(1,2)) {
  # fit <- arima(log(jj), c(p,d,q), seasonal = list(order = c(p,d,q), period = 12))
  fit <- sarima(dljj, p,d,q)
  # sum <- AIC(fit) + BIC(fit)
  sum <- fit$AIC + fit$BIC
  if (sum < min_sum) {
    min_sum <- sum
    final_p <- p
  }
}
final_p
sarima(log(jj), 2,1,0)   # the ACF of resuduals looks somewhat white (random)
sarima.for(log(jj), 10, 2,1,0)   # predict next 10 years, my common sense tells me somehting is wrong here...


# too busy recently, TO BE CONTINUED...
