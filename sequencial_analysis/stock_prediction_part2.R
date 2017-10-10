library(Quandl)
library(tidyverse)
library(tidyquant)
library(timetk)
library(forecast)
library(gridExtra)

Quandl.api_key("[YOUR Quandl API KEY]")   # change to your own Quandl API key

# download dataset
FXCB <- Quandl("WIKI/FXCB", collapse = "daily", start_date = "2015-04-10", type = "raw")
JPM <- Quandl("WIKI/JPM", collapse = "daily", start_date = "2015-04-10", type = "raw")
CCF <- Quandl("WIKI/CCF", collapse = "daily", start_date = "2015-04-10", type = "raw")
WFC <- Quandl("WIKI/WFC", collapse = "daily", start_date = "2015-04-10", type = "raw")
Citi <- Quandl("WIKI/C", collapse = "daily", start_date = "2015-04-10", type = "raw")

FXCB <- cbind(FXCB, Stock="FXCB")
JPM <- cbind(JPM, Stock="JPM")
CCF <- cbind(CCF, Stock="CCF")
WFC <- cbind(WFC, Stock="WFC")
Citi <- cbind(Citi, Stock="Citi")

us_stock_data <- rbind(FXCB, JPM, CCF, WFC, Citi)
us_stock_data$Date <- as.Date(us_stock_data$Date)

start_day <- ymd("2015-04-10")
end_day <- ymd("2016-04-10")

us_stock_data <- us_stock_data %>%
  tibble::as_tibble() %>%
  group_by(Stock)


# Visualizae the volatility of bank stocks
us_stock_data %>%filter(Stock=="JPM"|Stock=="Citi")%>%ggplot(aes(x=Date,y=Close))+
  geom_line(size=1)+
  geom_bbands(aes(high = High, low = Low, close = Close), ma_fun = SMA, sd=2,n = 20,size=0.75,
              color_ma = "royalblue4", color_bands = "red1")+
  coord_x_date(xlim = c(start_day, end_day), expand = TRUE)+
  facet_wrap(~ Stock, scales = "free_y")+
  labs(title = "Bollinger Band", x = "Date",y="Price") +
  theme(text = element_text(family = 'Gill Sans', color = "#444444",hjust=0.5)
        ,panel.background = element_rect(fill = 'lightyellow')
        ,panel.grid.minor = element_blank()
        ,panel.grid.major = element_blank()
        ,plot.title = element_text(size = 20,hjust=0.5,colour="orangered4")
        ,axis.title = element_text(size = 18, color = '#555555')
        ,axis.title.y = element_text(hjust=0.5,size=15)
        ,axis.title.x = element_text(hjust = 0.5,size=15)
  ) +
  theme(legend.position="none")
## both banks share similar trends and volatility through out the whole year

us_stock_data %>%filter(Stock!="Citi"&Stock!="JPM")%>%ggplot(aes(x=Date,y=Close))+
  geom_line(size=1)+
  geom_bbands(aes(high = High, low = Low, close = Close), ma_fun = SMA, sd=2,n = 20,size=0.75,
              color_ma = "royalblue4", color_bands = "red1")+
  coord_x_date(xlim = c(start_day, end_day), expand = TRUE)+
  facet_wrap(~ Stock, scales = "free_y")+
  labs(title = "Bollinger Band", x = "Date",y="Price") +
  theme(text = element_text(family = 'Gill Sans', color = "#444444",hjust=0.5)
        ,panel.background = element_rect(fill = 'lightyellow')
        ,panel.grid.minor = element_blank()
        ,panel.grid.major = element_blank()
        ,plot.title = element_text(size = 20,hjust=0.5,colour="orangered4")
        ,axis.title = element_text(size = 18, color = '#555555')
        ,axis.title.y = element_text(hjust=0.5,size=15)
        ,axis.title.x = element_text(hjust = 0.5,size=15)
  ) +
  theme(legend.position="none")
## WFC has larger volatility


# Predict Stock Prices
## Download monthly data
JPM <- Quandl("WIKI/JPM", collapse = "monthly", start_date = "2015-04-10", type = "raw")
Citi <- Quandl("WIKI/C", collapse = "monthly", start_date = "2015-04-10", type = "raw")

## Convert the data into df for regression model
JPM_df <- JPM
Citi_df <- Citi
colnames(JPM_df)<-c("Date","Open","High","Low","Close","Volume","Turnover")
colnames(Citi_df)<-c("Date","Open","High","Low","Close","Volume","Turnover")

## Change the scale of Volume
JPM_df$Volume <- JPM_df$Volume/100000
Citi_df$Volume <- Citi_df$Volume/100000

## Regression models
m1 <- lm(JPM_df$Close~JPM_df$High+JPM_df$Low+JPM_df$Volume)
p1_df <- as.data.frame(predict(m1,interval="predict"))

m3 <- lm(Citi_df$Close~Citi_df$High+Citi_df$Low+Citi_df$Volume)
p3_df <- as.data.frame(predict(m3, interval="predict"))

## Forecast using ARIMA to take out the seasonality and cyclic part of the stock
m2 <- arima(diff(JPM_df$Close),order=c(1,0,0))
m4 <- arima(diff(Citi_df$Close),order=c(1,0,0))
p2_df <- as.data.frame(predict(m2,n.ahead=3))
p4_df <- as.data.frame(predict(m4,n.ahead=3))

## Combining Regression and ARIMA together
p1_df <- p1_df[1:3,]
p1_df$fit <- p1_df$fit+p2_df$pred
p3_df <- p3_df[1:3,]
p3_df$fit <- p3_df$fit+p4_df$pred

## Create the date df for next three months
date_df <-as.data.frame(as.Date(c("2017-11-30","2017-12-31","2018-01-31")))
colnames(date_df) <- c("date")

## Modify the predict dataset and add variable "Key" 
p1_df <- cbind(p1_df,date_df)
p1_df["Key"] <- "Predicted"
p1_df <- p1_df[,c("date","fit","lwr","upr","Key")]  # reorder columns

## Modify the predict dataset for Axis and add variable "Key"
p3_df <- cbind(p3_df,date_df)
p3_df["Key"] <- "Predicted"
p3_df <- p3_df[,c("date","fit","lwr","upr","Key")]

## Rename the columns
colnames(p1_df)<-c("Date","Close","lwr","upr","Key")
colnames(p3_df)<-c("Date","Close","lwr","upr","Key")

## Modify the dataset
JPM_df <- JPM%>%select("Date","Close")
Citi_df <- Citi%>%select("Date","Close")

## Add two variable for confidence interval "lwr" and "upr"
var<-c("lwr","upr")

JPM_df[var]<-NA
Citi_df[var]<-NA

## Add the Key variable for Actual data
JPM_df["Key"]<-"Actual"
Citi_df["Key"]<-"Actual"

## Rbind the predicted and actual value for both of the Stocks
JPM_com <- rbind(JPM_df,p1_df)
JPM_com$Date<-as.Date(JPM_com$Date)

Citi_com <- rbind(Citi_df,p3_df)
Citi_com$Date<-as.Date(Citi_com$Date)

## Visualisation
JPM_Plot<-ggplot(data=JPM_com,aes(x= Date, y = Close,color=Key,label=Close)) +
  # Prediction intervals
  geom_ribbon(aes(ymin = lwr, ymax = upr, fill = Key), 
              fill = "khaki2", size = 0)+
  geom_line(size = 1.7) + 
  geom_point(size = 2)+
  labs(title = "Actual and Predicted Price, JPM", x = "Date",y="Price") +
  theme(text = element_text(family = 'Gill Sans', color = "#444444",hjust=0.5)
        ,panel.background = element_rect(fill = "honeydew")
        ,panel.grid.minor = element_blank()
        ,panel.grid.major = element_blank()
        ,plot.title = element_text(size = 20,hjust=0.5,colour="orangered4")
        ,axis.title = element_text(size = 18, color = '#555555')
        ,axis.title.y = element_text(hjust=0.5,size=15)
        ,axis.title.x = element_text(hjust = 0.5,size=15))

Citi_Plot<- ggplot(data=Citi_com,aes(x= Date, y = Close,color=Key,label=Close)) +
  # Prediction intervals
  geom_ribbon(aes(ymin = lwr, ymax = upr, fill = Key), 
              fill = "khaki2", size = 0)+
  geom_line(size = 1.7) + 
  geom_point(size = 2)+
  labs(title = "Actual and Predicted Price, Citi Bank", x = "Date",y="Price") +
  theme(text = element_text(family = 'Gill Sans', color = "#444444",hjust=0.5)
        ,panel.background = element_rect(fill = "honeydew")
        ,panel.grid.minor = element_blank()
        ,panel.grid.major = element_blank()
        ,plot.title = element_text(size = 20,hjust=0.5,colour="orangered4")
        ,axis.title = element_text(size = 18, color = '#555555')
        ,axis.title.y = element_text(hjust=0.5,size=15)
        ,axis.title.x = element_text(hjust = 0.5,size=15))

## Plots
grid.arrange(JPM_Plot,Citi_Plot,ncol = 1, nrow = 2)
