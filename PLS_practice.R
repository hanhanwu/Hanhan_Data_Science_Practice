path<- "[the root dir of your data sets]"
setwd(path)

# load data
train <- read.csv("PCA_train.csv")
test <- read.csv("PCA_test.csv")

# quick explore data
dim(train)
dim(test)
str(train)
str(test)
# check missisng data
table(is.na(train))
colSums(is.na(train))
summary(train)

#add a column
test$Item_Outlet_Sales <- 1

#combine the data set
combi <- rbind(train, test)

#impute missing values with median
combi$Item_Weight[is.na(combi$Item_Weight)] <- median(combi$Item_Weight, na.rm = TRUE)

# deal with abnormal values
##  1. Item_Visibility == 0 is impossible in practice 
combi$Item_Visibility <- ifelse(combi$Item_Visibility == 0,
                                median(combi$Item_Visibility), combi$Item_Visibility)

## 2. When there is missing categorical data, using "Other" to replace
levels(combi$Outlet_Size)
levels(combi$Outlet_Size)[1] <- "Other"
levels(combi$Outlet_Size)

levels(combi$Outlet_Identifier)
levels(combi$Item_Type)
levels(combi$Item_Identifier)

## 3. When there are mismatched data, standaridize them
library("plyr", lib.loc="/Library/Frameworks/R.framework/Versions/3.2/Resources/library")
combi$Item_Fat_Content <- revalue(combi$Item_Fat_Content,
                                  c("LF" = "Low Fat", "reg" = "Regular"))
combi$Item_Fat_Content <- revalue(combi$Item_Fat_Content, c("low fat" = "Low Fat"))
table(combi$Item_Fat_Content)

combi$Item_Outlet_Sales[is.na(combi$Item_Outlet_Sales)] <- median(combi$Item_Outlet_Sales, na.rm = TRUE)
levels(combi$Outlet_Type)[1] <- "Other"

# separate the response variables and predictors
pls_data <- combi[,c(2:6, 8:11, 1, 7, 12)]

# convert categorical data into numerical data
pls_data$Item_Type <- as.numeric(factor(pls_data$Item_Type, 
                                        labels=(1:length(levels(factor(pls_data$Item_Type))))))
pls_data$Item_Fat_Content <- as.numeric(factor(pls_data$Item_Fat_Content,
                                               labels=(1:length(levels(factor(pls_data$Item_Fat_Content))))))
pls_data$Outlet_Establishment_Year <- as.numeric(factor(pls_data$Outlet_Establishment_Year, 
                                                        labels=(1:length(levels(factor(pls_data$Outlet_Establishment_Year))))))
pls_data$Outlet_Location_Type <- as.numeric(factor(pls_data$Outlet_Location_Type, 
                                                   labels=(1:length(levels(factor(pls_data$Outlet_Location_Type))))))
pls_data$Outlet_Type <- as.numeric(factor(pls_data$Outlet_Type, labels=(1:length(levels(factor(pls_data$Outlet_Type))))))
pls_data$Item_Identifier <- as.numeric(factor(pls_data$Item_Identifier, 
                                                labels=(1:length(levels(factor(pls_data$Item_Identifier))))))
pls_data$Outlet_Identifier <- as.numeric(factor(pls_data$Outlet_Identifier, 
                                                  labels=(1:length(levels(factor(pls_data$Outlet_Identifier))))))
pls_data$Outlet_Size <- as.numeric(factor(pls_data$Outlet_Size, 
                                                labels=(1:length(levels(factor(pls_data$Outlet_Size))))))
str(pls_data)

##------------rescale to [0,1] range, OPTIONAL--------------##
new_pls_data <- pls_data
normal.tab<-tabulate(cut(normal.counts,breaks=seq(0,1,by=0.0001)))
for(i in names(new_pls_data)) {
  new_pls_data[[i]] <- rescale(dnorm(seq(min(pls_data[[i]]), max(pls_data[[i]])
                              ,length=length(new_pls_data[[i]])))
                               ,range(normal.tab))
}
##------------rescale to [0,1] range, OPTIONAL--------------##

library(plsdepot)

# plsreg2 is used when there are 1+ Ys
pls2 = plsreg2(new_pls_data[, 1:9], new_pls_data[, 10:12, drop = FALSE], comps = 3)
plot(pls2)
plot(new_pls_data[,10:12], pls2$y.pred)

# use plsreg1 and remove the 2 identifiers
pls_data1 <- subset(new_pls_data, select = -c(Item_Identifier, Outlet_Identifier))
pls1 = plsreg1(pls_data1[, 1:9], pls_data1[, 10, drop = FALSE], comps = 3)
plot(pls1)
plot(pls_data1$Item_Outlet_Sales, pls1$y.pred, type = "n", xlab="Original", ylab = "Predicted")
title("Comparison of responses", cex.main = 0.9)
abline(a = 0, b = 1, col = "gray85", lwd = 2)
text(pls_data1$Item_Outlet_Sales, pls1$y.pred, col = "#5592e3")
