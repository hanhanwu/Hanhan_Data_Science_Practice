# Deal with continuous variables
data <- data.frame(state.x77)
str(data)

library(ggplot2)
#plot Frost variable and check the data points are all over
qplot(y = Frost, data = data, colour = 'red')


# method 1 - create bins and add labels
bins <- cut(data$Frost, 3, include.lowest = TRUE)
bins
bins <- cut(data$Frost, 3, include.lowest = TRUE, labels = c('Low', 'Medium', 'High'))
bins


# method 2 - normalization, Z-score
## I like this z-score tutorial: http://stats.seandolinar.com/calculating-z-scores-with-r/
income <- matrix(data$Income)
hist(income)
income_sd <- sd(income)*sqrt((length(income)-1)/length(income))
income_mean <- mean(income)
get_z <- function(v, my_mean, my_sd) (v-my_mean)/my_sd
apply(income, 1, get_z, income_mean, income_sd)


# method 3 - convert highly skewed variables
## check skewness 
library(e1071)
## in this case, the value indiaates its skewed towards right
skewness(data$Income)
skewness(data$Illiteracy)  
## or simply check histogram
hist(data$Income)
hist(data$Illiteracy)
## data$Illiteracy is highly skewed, change to log values, much better!
hist(log(data$Illiteracy))
skewness(log(data$Illiteracy))
