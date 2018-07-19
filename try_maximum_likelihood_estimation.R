# try MLE (maximum likelihood estimaiton) to predict hourly ticket selling count
## Download the input data from https://s3-ap-south-1.amazonaws.com/av-blog-media/wp-content/uploads/2018/07/Train_Tickets.csv

path<- "[YOUR INPUT FOLDER PATH]"  # Use the folder path that stores the input file
setwd(path)

raw_data <- read.csv("MLE_tickets_sold.csv")
head(raw_data)

hist(raw_data$Count, breaks = 50,probability = T ,main = "Histogram of Count Variable")
lines(density(raw_data$Count), col="red", lwd=2)
# The distribution can be treated as Poisson Distribution:
## a discrete probability distribution that 
## expresses the probability of a given number of events 
## occurring in a fixed interval of time or space if these events occur with a 
## known constant rate and independently of the time since the last event.

