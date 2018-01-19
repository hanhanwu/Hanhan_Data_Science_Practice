library(RODBC)
library(data.table)
require(data.table)
library(dplyr)
library(caret)
library(dummies)
library(ggplot2)
library(plotly)
library(FSelector)
library('e1071')
library(mlr)
library(ROSE)

# Check Overall Missing Value Percentage
## Altough mlr package will show you number of missing values for each col, it's not percentage
## Here, will show all the missing value percentage
missing_percentage <- sapply(my_data, function(x){sum(is.na(x))/length(x)})*100
sort(missing_percentage, decreasing = T)


# Univarite Analysis - check categorical data distribution
fact_distribution_plot <- function(a){
  counts <- table(a)
  barplot(counts)
}

    ## Chekc whether it's normal distribution
    par(mar = rep(2, 4))        # set up the graphics frame
    hist(dljj, prob=TRUE, 12)   # histogram    
    lines(density(data))     # smooth it
    qqnorm(data)             # normal Q-Q plot  
    qqline(data)             # add a line 

# Univarite Analysis - check numerical data distribution
num_distribution_plot <- function(a, q){
  ggplot(data = q, aes(x= a, y=..density..)) + geom_histogram(fill="blue",color="red",alpha = 0.5,bins =100) + geom_density()
  ggplotly()
}

# Bi-varite Analysis on numerical variable - polt multiple plots together
multiplot <- function(..., plotlist=NULL, file, cols=1, layout=NULL) {
  library(grid)
  
  # Make a list from the ... arguments and plotlist
  plots <- c(list(...), plotlist)
  
  numPlots = length(plots)
  
  # If layout is NULL, then use 'cols' to determine layout
  if (is.null(layout)) {
    # Make the panel
    # ncol: Number of columns of plots
    # nrow: Number of rows needed, calculated from # of cols
    layout <- matrix(seq(1, cols * ceiling(numPlots/cols)),
                     ncol = cols, nrow = ceiling(numPlots/cols))
  }
  
  if (numPlots==1) {
    print(plots[[1]])
    
  } else {
    # Set up the page
    grid.newpage()
    pushViewport(viewport(layout = grid.layout(nrow(layout), ncol(layout))))
    
    # Make each plot, in the correct location
    for (i in 1:numPlots) {
      # Get the i,j matrix positions of the regions that contain this subplot
      matchidx <- as.data.frame(which(layout == i, arr.ind = TRUE))
      
      print(plots[[i]], vp = viewport(layout.pos.row = matchidx$row,
                                      layout.pos.col = matchidx$col))
    }
  }
}


# Bi-varite Analysis on categorical variable - polt multiple plots together
p1 <- rose_scaled_data$module_documented[which(rose_scaled_data$IsUnderrated=='Y')]
pt1 <- ggplot(data=data.frame(p1), aes(x= p1, y=..density..)) + geom_histogram(fill="blue",color="red",alpha = 0.5,bins =100) + geom_density()
p2 <- rose_scaled_data$module_documented[which(rose_scaled_data$IsUnderrated=='N')]
pt2 <- ggplot(data=data.frame(p2), aes(x= p2, y=..density..)) + geom_histogram(fill="blue",color="red",alpha = 0.5,bins =100) + geom_density()
multiplot(pt1, pt2, cols=2)


# seperate data into numerical data and categorical data [operate on columns]
origin_nums <- sapply(q1, is.numeric)
origin_num_data <- subset(q1, select = origin_nums==T)
origin_fact_data <- subset(q1, select = origin_nums==F)
# seperate data based on row values
subset(q1, col1=='Ice-cream' & is.na(col2)==T) # WARNING: this may choose only part of the data, which() maybe better


# explore the smaller dataset, in this case, the categorical data
  ## check missing value percentage
  NA_perct <- sapply(origin_fact_data, function(x){sum(is.na(x))/length(x)})*100
  NA_perct
  ## check NA percentage for each group, in this case group by "my_group", apply to all others columns using ".~"
  match_result_NA_perct <- aggregate(.~ my_group, data=my_data, 
                                   function(x) {sum(is.na(x))/length(x)*100}, na.action = NULL)

  ## univariate analysis
  fact_distribution_plot(origin_fact_data$PaymentType)

  ## bivariate analysis, check the feature with the target
  ggplot(origin_fact_data, aes(HasWriteOff, ..count..)) + geom_bar(aes(fill = PaymentType), position = "dodge") 


# explore numerical data
  ## check missing value percentage
  NA_perct <- sapply(origin_num_data, function(x){sum(is.na(x))/length(x)})*100
  NA_perct 

  ## convert to factor without losing the original values  
  ### [but this may not be necessary if you don't have any other operation on this feature]
  origin_num_data$BranchID <- as.factor(as.character(origin_num_data$BranchID))


  ## deal with data skewness
  num_distribution_plot(log(origin_num_data$LoanAmount), origin_num_data)
  num_distribution_plot(origin_num_data$mlbf_InterestRate, origin_num_data)
  origin_num_data$ApprovedAmount <- log(origin_num_data$LoanAmount)

  ## to know central tendency and outliers with numbers
  quantile(origin_num_data$LoanAmount)
  boxplot(origin_num_data$LoanAmount)

  ## to check numerical feature distribution based on the target
  p1 <- origin_num_data$LoanAmount[which(origin_num_data$HasWriteOff==1)]
  pt1 <- ggplot(data=data.frame(p1), aes(x= p1, y=..density..)) + geom_histogram(fill="blue",color="red",alpha = 0.5,bins =100) + geom_density()
  p2 <- origin_num_data$LoanAmount[which(origin_num_data$HasWriteOff==0)]
  pt2 <- ggplot(data=data.frame(p2), aes(x= p2, y=..density..)) + geom_histogram(fill="blue",color="red",alpha = 0.5,bins =100) + geom_density()
  multiplot(pt1, pt2, cols=2)


# put all the final categorical and numerical data together
q2 <- cbind(fact_data, num_data)


# Check feature variance
## you have to impute all the missing values before checking variance
## Here, convert all the categorical data into numerical data
for (i in seq_along(dm_data)) set(dm_data, i=which(is.na(dm_data[[i]])), j=i, value="MISSING")
## convert categorical col into numerical
dm_data[, num_feature:=as.integer(dm_data$cat_feature)]

## Numerical data
num_feature_variance <- data.table(group_by(dm_data, color_group) %>%   # change color_group to your group by col
                                                    summarise(GroupVariance=var(rep(num_feature))))
num_feature_variance # variance for groups in this feature
var(same_user_sebset$num_feature) # variance for this feature as a whole

## Categorical data
dm_data[, cat_feature:=as.integer(as.factor(dm_data$cat_feature))]
cat_feature_variance <- data.table(group_by(dm_data, color_group) %>%  # change color_group to your group by col
                                  summarise(GroupVariance=var(rep(cat_feature))))
cat_feature_variance       
var(as.integer(as.factor(dm_data$cat_feature))) 



