library(splines)
library(ISLR)
attach(Wage)

summary(Wage)
agelims <- range(age)
age.grid <- seq(from=agelims[1], to=agelims[2])
age.grid

# fit wage to age through a regression spline
fit <- lm(wage~bs(age, knots = c(25, 40, 60)), data=Wage)
pred <- predict(fit, newdata = list(age=age.grid),se=T)
pred

plot(age, wage, col='pink')
lines(age.grid, pred$fit, lwd=2)
lines(age.grid, pred$fit+2*pred$se, lty="dashed", col='purple')
lines(age.grid, pred$fit-2*pred$se, lty="dashed", col='green')

dim(bs(age, knots = c(25, 40, 60)))  # 6 basis functions
dim(bs(age, df=6))  # a cubic spline with 3 knots has 6 degree of freedom
attr(bs(age, df=6), "knots")  # 25%, 50%, 70% indicates the percentile values of age

# fit a natural spline, use ns()
fit2 <- lm(wage~ns(age,df=4), data = Wage)  # degree of freedom=4
pred2 <- predict(fit2, newdata = list(age=age.grid), se=T)
lines(age.grid, pred2$fit, col='red', lwd=2)

# fit a smoothing spline
plot(age, wage, xlim = agelims, cex=0.5, col='pink')
title("Smoothing Spline")
fit <- smooth.spline(age, wage, df=16)
fit2 <- smooth.spline(x=age, wage, cv=TRUE)
fit2$df  # the corss validation yields 6.8 degree of freedom
lines(fit, col='purple', lwd=2)
lines(fit2, col='green', lwd=2)
legend("topright", legend = c("16 DF", "6.8 DF"), col = c("purple", "green"), lty = 1, lwd = 2, cex = 0.7)

# local regression, it's not spline, but I want to try
plot(age, wage, xlim = agelims, ces=0.5, col='grey')
title("Local Regression")
fit <- loess(wage~age, span = 0.2, data = Wage)
fit2 <- loess(wage~age, span = 0.5, data = Wage)
lines(age.grid, predict(fit, data.frame(age=age.grid)), col="purple", lwd=2)
lines(age.grid, predict(fit2, data.frame(age=age.grid)), col="green", lwd=2)
legend("topright", legend = c("Span=0.2", "Span=0.5"), col=c("purple", "green"), lty=1, lwd=2, cex= 0.7)
