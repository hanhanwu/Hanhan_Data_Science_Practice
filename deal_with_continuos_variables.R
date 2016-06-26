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
