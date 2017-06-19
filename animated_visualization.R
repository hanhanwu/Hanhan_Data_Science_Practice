# Created Animated Visualization
# export MAGICK_HOME="[the folder of you ImageMagick]/ImageMagick-7.0.5"
# export PATH="$MAGICK_HOME/bin:$PATH"

library(plyr)
library(dplyr)
library(ggmap)
library(ggplot2)
library(devtools)
install_github("dgrtwo/gganimate")
library(gganimate)


path<- "[the path of your data]"   # change path here
setwd(path)

earthquare_data <- read.csv("earthquake_data.csv", stringsAsFactors = F)
names(earthquare_data)
dim(earthquare_data)
earthquare_data<-earthquare_data%>%filter(Magnitude >= 7)   # only choose those with more than 7 magnitude
dim(earthquare_data)
head(earthquare_data)

# Split date into month, day, year
earthquare_data$Date <- as.character(earthquare_data$Date)
list <- strsplit(earthquare_data$Date, "/")
eq_date1 <- ldply(list)
colnames(eq_date1) <- c("Month", "Day", "Year")
head(eq_date1)
earthquare_data <- cbind(earthquare_data, eq_date1)
head(earthquare_data)

world_data <- map_data("world")
head(world_data)
levels(as.factor(world_data$region))
world_data <- world_data[world_data$region != "Antarctica",]
map <- ggplot() + geom_map(data=world_data, map=world_data, aes(x=long, y=lat, map_id=region), color='#333300',fill='#663300')
p <- map + geom_point(data = earthquare_data, aes(x = Longitude, y = Latitude, 
                                     frame = Year, 
                                     cumulative = TRUE,size=earthquare_data$Magnitude), alpha = 0.3, 
                      size = 2.5,color="#336600")+
  geom_jitter(width = 0.1) +labs(title = "Earthquake above 7 point on richter scale")+theme_void()
gganimate(p)
