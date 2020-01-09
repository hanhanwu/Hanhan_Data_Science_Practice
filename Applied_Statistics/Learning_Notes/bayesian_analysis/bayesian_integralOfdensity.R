mean_val <- 0.0
std_val <- 0.2
xlow <- mean_val - 3*std_val  # lower end of x-axis
xhigh <- mean_val + 3*std_val # higher end of x-axis
dx <- 0.02  # interval width on x-axis

# specify comb points along x-axis
x <- seq(from=xlow, to=xhigh, by=dx)
x

# compute y values, (probability density at each value of x), PDF of normal distribution
y <- (1/(std_val*sqrt(2*pi))) * exp(-0.5 * ((x - mean_val)/std_val)^2)
y

# plot the distribution
plot(x, y, type='h', lwd=1, ces.axis=1.5,
     xlab='x', ylab = 'p(x)', main='Normal Probability Density')
lines(x, y)  ## add the curve

# approximate the integral as the sum of width * height of each interval
area <- sum(dx*y)
area

# add text into the chart
text(-std_val , 0.9*max(y) , bquote( paste(mu ," = " ,.(mean_val))), adj=c(1,0.5))
text(-std_val , 0.8*max(y) , bquote( paste(sigma ," = " ,.(std_val))), adj=c(1,0.5))
text(std_val , 0.9*max(y) , bquote( paste(Delta ,"x = " ,.(dx))), adj=c(0,0.5))
text(std_val , 0.8*max(y) , bquote( paste(sum(,x,) ," ", Delta, "x p(x)", .(signif(area, 3)))), adj=c(0,0.5))

# save plot to ESP file
dev.copy2eps(file = "IntegralOfDensity.eps")
