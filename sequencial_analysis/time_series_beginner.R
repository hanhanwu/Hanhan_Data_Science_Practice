library(astsa)
data()  # load all the data in package astsa

dim(jj)
nrow(jj)
ncol(jj)


jjm <- as.matrix(jj)
dim(jjm)
head(jj)

# normal distribution for 99 random numbers
# start from 77th row, April
# display 12 months
(zardoz = ts(rnorm(99), start=c(77,4), frequency=12))   
