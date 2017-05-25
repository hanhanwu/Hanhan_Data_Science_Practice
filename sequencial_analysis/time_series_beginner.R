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


# too busy recently, TO BE CONTINUED...
