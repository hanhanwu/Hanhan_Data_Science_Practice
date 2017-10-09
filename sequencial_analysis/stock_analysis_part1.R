library(Quandl)
library(tidyverse)
library(ggplot2)
library(tidyquant)
library(timetk)
library(forcats)
library(stringr)
devtools::install_github("dgrtwo/gganimate")
library(gganimate)
library(plyr)
library(stringr)
library(gridExtra)

Quandl.api_key("[YOUR Quandl API]")  # change to your API key here

# download dataset
FXCB <- Quandl("WIKI/FXCB", collapse = "daily", start_date = "2015-04-10", type = "raw")
head(FXCB)
min(FXCB$Date)
max(FXCB$Date)
JPM <- Quandl("WIKI/JPM", collapse = "daily", start_date = "2015-04-10", type = "raw")
min(JPM$Date)
max(JPM$Date)
CCF <- Quandl("WIKI/CCF", collapse = "daily", start_date = "2015-04-10", type = "raw")
min(CCF$Date)
max(CCF$Date)
WFC <- Quandl("WIKI/WFC", collapse = "daily", start_date = "2015-04-10", type = "raw")
min(WFC$Date)
max(WFC$Date)
Citi <- Quandl("WIKI/C", collapse = "daily", start_date = "2015-04-10", type = "raw")
min(Citi$Date)
max(Citi$Date)

FXCB <- cbind(FXCB, Stock="FXCB")
head(FXCB)
JPM <- cbind(JPM, Stock="JPM")
CCF <- cbind(CCF, Stock="CCF")
WFC <- cbind(WFC, Stock="WFC")
Citi <- cbind(Citi, Stock="Citi")

us_stock_data <- rbind(FXCB, JPM, CCF, WFC, Citi)
summary(us_stock_data)


# visualize monthly prices
## ggplot theme functions: https://www.rdocumentation.org/packages/ggplot2/versions/2.2.1/topics/theme
us_stock_data$Date <- as.character(us_stock_data$Date)
## add Year, Month, Date as new columns
list <- strsplit(us_stock_data$Date, "-")
df_Date <- ldply(list)
colnames(df_Date) <- c("Year", "Month", "Day")

us_stock_data <- cbind(us_stock_data, df_Date)
head(us_stock_data)
us_stock_data$Date <- as.Date(us_stock_data$Date)

summary(us_stock_data$Volume)
us_stock_data$Volume <- us_stock_data$Volume/100000

P<- ggplot(us_stock_data,aes(factor(Stock),Close,color=Stock,frame=Month)) +
  geom_jitter(aes(size = Close, colour=Stock, alpha=.02)) +
  ylim(0,1000)+
  labs(title = "US Banks Stock Monthly Prices [From 2015]", x = "Banks", y= "Close Price") +
  theme(panel.border = element_blank(),
        panel.grid.major = element_line(colour = "grey61", size = 0.5, linetype = "dotted"),
        panel.grid.minor = element_blank(),
        axis.line=element_line(colour="black"),
        plot.title = element_text(hjust = 0.5,size=18,colour="indianred4"))+
  theme(legend.position="none")

gganimate(P) # this is an animation


# Daily Close Price for each Bank
us_stock_data<-us_stock_data%>%
  tibble::as.tibble()%>%
  group_by(Stock)

## Visualisation for Daily Stock Prices
us_stock_data %>%
  ggplot(aes(x = Date, y = Close, color = Stock)) +
  geom_point() +
  labs(title = "Daily Close Price", x = "Month",y="Close Price") +
  facet_wrap(~ Stock, ncol = 3, scale = "free_y") +
  scale_fill_tq(fill="green4",theme="light") +
  theme_tq() +
  theme(panel.border = element_blank(),
        panel.grid.major = element_line(colour = "grey61", size = 0.5, linetype = "dotted"),
        panel.grid.minor = element_blank(),
        axis.line=element_line(colour="black"),
        plot.title = element_text(hjust = 0.5,size=18,colour="indianred4"))+
  theme(legend.position="none")


# Relationship between Volume and Close Price
E<-us_stock_data %>%
  ggplot(aes(x = Volume, y = Close, color = Stock,frame=Month)) +
  geom_smooth(method='loess') +
  xlim(0,400)+
  labs(title = "Monthly Volume vs Price", x = "Volume (Lacs)",y="Close Price") +
  facet_wrap(~ Stock, ncol = 3, scale = "free_y") +
  scale_fill_tq(fill="green4",theme="light") +
  theme_tq() +
  theme(panel.border = element_blank(), 
        panel.grid.major = element_blank(),
        panel.grid.minor = element_blank(),
        plot.title = element_text(hjust = 0.5,size=18,colour="indianred4"),
        axis.line = element_line(colour = "black"))+
  theme(legend.position="none")

gganimate(E,ani.width=600,ani.height=400,interval=0.7)


# Finding the Density Distribution of Deviation of High Price from Open Price
## Weekly deviating
## transmute_tq() for get weekly price
## tq_transmute() for weekly average of differences
us_stock_data_High<-us_stock_data%>%mutate(Dev_High=High-Open) # deviation
us_stock_data_Low<-us_stock_data%>%mutate(Dev_Low=Open-Low)
## High price weekly average
us_stock_data_High_Week <- us_stock_data_High %>%
  tq_transmute(
    select     = Dev_High,
    mutate_fun = apply.weekly, 
    FUN        = mean,
    na.rm      = TRUE,
    col_rename = "Dev_High_Mean"
  )
## Low price weekly average
us_stock_data_Low_Week <- us_stock_data_Low %>%
  tq_transmute(
    select     = Dev_Low,
    mutate_fun = apply.weekly, 
    FUN        = mean,
    na.rm      = TRUE,
    col_rename = "Dev_Low_Mean"
  )
## Visualize Density Distribution of High Price
High<-us_stock_data_High_Week%>%ggplot(aes(x=Dev_High_Mean,color=Stock))+
  geom_dotplot(binwidth=0.50,aes(fill=Stock))+
  xlim(0,10)+
  scale_fill_manual(values=c("#999999", "#E69F00","#CC9933","#99FF00","#CC3399","#FF9933"))+
  labs(title="Distribution of High Price Deviation from Open Price",x="Weekly Mean Deviation")+
  facet_wrap(~Stock,ncol=3,scale="free_y")+
  scale_color_tq(values=c("#999999"))+
  theme_tq()+
  theme(panel.border = element_blank(),
        panel.grid.major = element_line(colour = "grey61", size = 0.5, linetype = "dotted"),
        panel.grid.minor = element_blank(),
        axis.line=element_line(colour="black"),
        plot.title = element_text(hjust = 0.5,size=16,colour="indianred4"))+
  theme(legend.position="none")

Low<-us_stock_data_Low_Week%>%ggplot(aes(x=Dev_Low_Mean,color=Stock))+
  geom_dotplot(binwidth=0.50,aes(fill=Stock))+
  xlim(0,10)+
  scale_fill_manual(values=c("#999999", "#E69F00","#CC9933","#99FF00","#CC3399","#FF9933"))+
  labs(title="Distribution of Weekly Low Price Deviation from Open Price",x="Weekly Mean Deviation")+
  facet_wrap(~Stock,ncol=3,scale="free_y")+
  scale_color_tq(values=c("#999999"))+
  theme_tq()+
  theme(panel.border = element_blank(),
        panel.grid.major = element_line(colour = "grey61", size = 0.5, linetype = "dotted"),
        panel.grid.minor = element_blank(),
        axis.line=element_line(colour="black"),
        plot.title = element_text(hjust = 0.5,size=16,colour="indianred4"))+
  theme(legend.position="none")

grid.arrange(High,Low,ncol = 2, nrow = 1)


# Autocorrelation Lags
k <- 1:180   # k is the lag, here let's check the lag of 410 days
col_names <- paste0("lag_", k)
## Only Select Columns "Date" and "Close" from hte master data frame.
us_stock_data_lags<-us_stock_data%>%
  tibble::as_tibble() %>%
  group_by(Stock)
us_stock_data_lags<-us_stock_data_lags%>%select(Date,Close)
## Apply lag.xts function using tq_mutate
us_stock_data_lags<-us_stock_data_lags%>%
  tq_mutate(
    select = Close,
    mutate_fun = lag.xts,
    k=1:180,
    col_rename=col_names
  )
## Calculate the autocorrelations and 95% cutoffs
us_stock_data_AutoCorrelations<-us_stock_data_lags %>%
  gather(key = "lag", value = "lag_value", -c(Stock,Date, Close)) %>%
  mutate(lag = str_sub(lag, start = 5) %>% as.numeric) %>%
  group_by(Stock, lag) %>%
  dplyr::summarize(
    cor = cor(x = Close, y = lag_value, use = "pairwise.complete.obs"),
    cutoff_upper = 2/(n())^0.5,
    cutoff_lower = -2/(n())^0.5
  )
## Visualisation of Autocorrelation: ACF Plot
us_stock_data_AutoCorrelations %>%
  ggplot(aes(x = lag, y = cor, color = Stock, group = Stock)) +
  ## Add horizontal line a y=0
  geom_hline(yintercept = 0) +
  ## Plot autocorrelations
  geom_point(size = 2) +
  geom_segment(aes(xend = lag, yend = 0), size = 1) +
  ## Add cutoffs
  geom_line(aes(y = cutoff_upper), color = "blue", linetype = 2) +
  geom_line(aes(y = cutoff_lower), color = "blue", linetype = 2) +
  ## Add facets
  facet_wrap(~ Stock, ncol = 3) +
  ##  Aesthetics
  expand_limits(y = c(-1, 1)) +
  scale_color_tq() +
  theme_tq() +
  labs(
    title = paste0("Tidyverse ACF Plot: Lags ", rlang::expr_text(k)),
    x = "Lags"
  ) +
  theme(
    legend.position = "none",
    axis.text.x = element_text(angle = 45, hjust = 1),
    panel.grid.major = element_line(colour = "grey61", size = 0.5, linetype = "dotted"),
    plot.title = element_text(hjust = 0.5,size=18,colour="indianred4")
  )
