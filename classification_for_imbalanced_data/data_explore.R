
# Univarite Analysis - check categorical data distribution
fact_distribution_plot <- function(a){
  counts <- table(a)
  barplot(counts)
}

# Univarite Analysis - check numerical data distribution
num_distribution_plot <- function(a, q){
  ggplot(data = q, aes(x= a, y=..density..)) + geom_histogram(fill="blue",color="red",alpha = 0.5,bins =100) + geom_density()
  ggplotly()
}

# Bi-varite Analysis - polt multiple plots together
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


# seperate data into numerical data and categorical data
origin_nums <- sapply(q1, is.numeric)
origin_num_data <- subset(q1, select = origin_nums==T)
origin_fact_data <- subset(q1, select = origin_nums==F)


# explore the smaller dataset, in this case, the categorical data
  ## check missing value percentage
  NA_perct <- sapply(origin_fact_data, function(x){sum(is.na(x))/length(x)})*100
  NA_perct

  ## univariate analysis
  fact_distribution_plot(origin_fact_data$PaymentType)

  ## bivariate analysis, check the feature with the target
  ggplot(origin_fact_data, aes(HasWriteOff, ..count..)) + geom_bar(aes(fill = PaymentType), position = "dodge") 


# explore numerical data
  ## check missing value percentage
  NA_perct <- sapply(origin_num_data, function(x){sum(is.na(x))/length(x)})*100
  NA_perct 

## convert to factor without losing the original values
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


## put all the final categorical and numerical data together
q2 <- cbind(fact_data, num_data)
