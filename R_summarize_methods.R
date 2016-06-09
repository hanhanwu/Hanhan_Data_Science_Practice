# apply() - apply the method to the indicated margin (row or column)
m <- matrix(c(1:10, 11:20), nrow = 10, ncol = 2)
m
apply(m,1,mean)  # margin=1 indicate apply the method to rows
apply(m,2,mean)  # margin=2 indicates apply the method to columns


# lapply() - apply the method to each element in a list, return the same length of list as the origional list
l <- list(a = 1:10, b = 11:20)
l
lapply(l, mean)


# sapply() - does the same thing as lapply(), but returns a vector/matrix instead of a list
l <- list(a = 1:10, b = 11:20)
l
s <- sapply(l, mean)
s


# tapply() - tapply(X, INDEX, FUN), X is a vector
attach(iris)
summary(iris)
tapply(iris$Petal.Length, Species, mean)


# by() - an object-oriented wrapper for ‘tapply’ applied to data frames
by(iris[,1:4], Species, colMeans)


# sqldf() - oh! Everyone loves R! It can do something like sql
install.packages(sqldf)
library(sqldf)
summarization <- sqldf('select Species, avg("Petal.Length") `Petal.Length_mean` from iris where Species is not null group by Species')
summarization


# ddply() - same as sqldf
library(plyr)
ddply(iris, "Species", summarise, Petal.Length_mean = mean(Petal.Length))
