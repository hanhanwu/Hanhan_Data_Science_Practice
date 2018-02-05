# Using k-ordered Markov Chains and Heuristics Model to find 
## which channels are important for final conversion

library(devtools)
install_github("cran/ChannelAttribution")  # you may need this all the time

library(Rcpp)
library(RcppArmadillo)
library(ChannelAttribution)
library(ggplot2)
library(reshape)
library(dplyr)
library(plyr)
library(reshape2)
library(markovchain)
library(plotly)

path<- "/Users/hanhanwu/Desktop/"
setwd(path)

raw_data <- read.csv("[your csv file path]")  # change the csv path here
head(raw_data)

# Channel 20 - A customer has decided which to buy
# Channel 21 - A customer has made the purchase (Conversion Point!!)
# Channel 22 - A customer has not decided yet

# Channel 1,2,3 - Company website
# Channel 4,5,6,7,8 - Industry Adveisory Research Reports
# Channel 9,10 - Organic Searches, Forums, Online Reviews
# Channel 11 - Price Comparison Aggregators
# Channel 12,13 - Social Network, Friends
# Channel 14 - Expert online/offline
# Channel 15,16,17 - Retail Stores
# Channel 18,19 - Other

for(row in 1:nrow(raw_data)){
  if(21 %in% raw_data[row,]){
    raw_data$convert[row] = 1
  }
}

head(raw_data)
cols <- colnames(raw_data)
raw_data$path <- do.call(paste, c(raw_data[cols], sep=">"))  # do.call method call a function with a list of arguments
head(raw_data)
strsplit(raw_data$path[1], ">21")[[1]][1]

for(row in 1:nrow(raw_data)){
  raw_data$path[row] = strsplit(raw_data$path[row], ">21")[[1]][1]
}

# only choose the path and conversion
channel_paths <- raw_data[,c(23,22)]  # better in this order
head(channel_paths)
# Summarize the number of conversion for the same paths
channel_paths <- ddply(channel_paths, ~path, summarise, conversion=sum(convert))
head(channel_paths)
summary(channel_paths$conversion)

# Heuristic Model
H <- heuristic_models(Data=channel_paths, var_path='path', var_conv='conversion', var_value='conversion')
H

# Here only use 1 order Markov Chain
k=1
M <- markov_model(Data=channel_paths, 'path', 'conversion', var_value='conversion', order = k)
M

HM <- merge(H, M, by='channel_name')
head(HM)

select_cols <- c("channel_name", "first_touch_conversions", 
                 "last_touch_conversions", "linear_touch_conversions", "total_conversion")
select_HM <- HM[,select_cols]
head(select_HM)

select_HM <- melt(select_HM, id='channel_name')
head(select_HM)
summary(select_HM$channel_name)

# plot which channels get higher conversion values
ggplot(select_HM, aes(channel_name, value, fill = variable)) +
  geom_bar(stat='identity', position='dodge') +
  ggtitle('Channel conversions') +
  theme(axis.title.x = element_text(vjust = -2)) +
  theme(axis.title.y = element_text(vjust = +2)) +
  theme(title = element_text(size = 16)) +
  theme(plot.title=element_text(size = 20)) +
  ylab("")
