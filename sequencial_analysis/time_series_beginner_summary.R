library(astsa)
data()

# Part 1 - Time Series Simulation
set.seed(7799)
## AR1 (set p=1)
x1 = arima.sim(list(order=c(1,0,0), ar=.9), n=100) 
x2 = arima.sim(list(order=c(1,0,0), ar=-.9), n=100)
par(mfrow=c(2,1))
plot(x1, main=(expression(AR(1)~~~phi==+.9)))  # ~ is a space and == is equal  
plot(x2, main=(expression(AR(1)~~~phi==-.9)))
dev.new()           # open another graphics device if you wish
acf2(x1, 20)
dev.new()           
acf2(x2, 20)

## AR2 (set p=2)
x = arima.sim(list(order=c(2,0,0), ar=c(1,-.9)), n=100) 
plot(x, main=(expression(AR(2)~~~phi[1]==1~~~phi[2]==-.9)))
acf2(x)

## MA1 (set q=1)
x = arima.sim(list(order=c(0,0,1), ma=.8), n=100)
plot(x, main=(expression(MA(1)~~~theta==.8)))
acf2(x)

## ARIMA(1,1,1)
x = arima.sim(list(order=c(1,1,1), ar=.9, ma=-.5), n=200)
plot(x, main=(expression(ARIMA(1,1,1)~~~phi==.9~~~theta==-.5)))
acf2(x, 30)


# Part 2 - JJ data (observation, fit model & predict)

time(jj)
cycle(jj)
## plot the data
par(mfrow = c(2,1))
plot(jj, ylab="Earnings per Share", main="J & J")
ts.plot(jj, ylab="Earnings per Share", main="J $ J")
## plot the data and its gorth rate
plot(diff(jj), main="diffed")
plot(diff(log(jj)), main="logged and diffed")  # the visualization here tells us, for jj data, it needs at least differencing and logging to convert to stationary
## check whether dljj is in fact stationary
dljj <- diff(log(jj))
acf2(dljj)    # It should be an AR model

## Fitting Method 1 - Try to fit the regression
Q <- factor(cycle(dljj))
trend <- time(dljj) - 1970 # center data
reg <- lm(dljj~ 0 + trend + Q, na.action = NULL)
options(show.signif.stars=FALSE)
summary(reg)
plot(dljj, type="o")
lines(fitted(reg), col=2)  # looks really fit...
par(mfrow=c(2,1))
plot(resid(reg))
acf(resid(reg), 20)   # but the residuals do not look white....

## Fitting Method 2 - Find optimal (p,d,q) to build the model and predict time series
final_p <- 1
d <- 1
min_sum <- 0
final_q <- 1
for (p in c(1.2)) {
  for (q in c(1,2)) {
    # fit <- arima(dljj, c(p,d,q), seasonal = list(order = c(p,d,q), period = 12))
    fit <- sarima(dljj, p,d,q)
    # sum <- AIC(fit) + BIC(fit)
    sum <- fit$AIC + fit$BIC
    if (sum < min_sum) {
      min_sum <- sum
      final_q <- q
      final_p <- p
    }
  }
}

final_q
final_p

sarima(dljj, 1.2,1,2)   # the ACF of resuduals looks somewhat white (random)
sarima.for(dljj, 7, 2,1,0)

