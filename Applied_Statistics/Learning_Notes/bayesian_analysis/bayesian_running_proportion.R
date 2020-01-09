# generate 500 records of fair coin flips, 1 means head, 0 means tail
N <- 500
set.seed(410)
flipsequence <- sample(x=c(0,1), prob = c(0.5, 0.5), size = N, replace = TRUE)
summary(flipsequence)

# cumulated sum
r <- cumsum(flipsequence)
r

# running proportion = cumulated sum / cumulated count
n <- 1:N
running_prop <- r/n 
running_prop

# plot running propotions of heads
plot(n, running_prop, type = 'o', log = 'x',
     xlim = c(1,N), ylim = c(0.0, 1.0), cex.axis=1.5,
     xlab='Flip Number', ylab = 'Heads Proportion',
     main = 'Running Proportion of Heads')
lines(c(1,N), c(0.5, 0.5), lty=3)  # add horizontal line
## add text in the chart, by showing the first 10 coint flips results
flipletters = paste( c("T","H")[ flipsequence[ 1:10 ] + 1 ] , collapse="" )
displaystring = paste( "Flip Sequence = " , flipletters , "..." , sep="" )
text( 5 , .9 , displaystring , adj=c(0,1) , cex=1.3 )
## add text in the chart, by showing the last running proportion
text( N , .3 , paste("End Proportion =",running_prop[N]) , adj=c(1,0) , cex=1.3 )

# save the plot as an EPS file
path<- "/Users/hanhanwu/Desktop/R/"
setwd(path)
dev.copy2eps(file='running_proportion.eps')
