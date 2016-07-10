path<- "[your data folder path]"
setwd(path)

train <- read.csv("blood_train.csv", header = TRUE)
str(train)
test <- read.csv("blood_test.csv", header = TRUE)
str(test)

library("dplyr")
# volumn is closely correlated to the number of donations, remove it
# remove ID too
train <- select(train, -c(X, Total.Volume.Donated..c.c..))
str(train)
test <- select(test, -c(X, Total.Volume.Donated..c.c..))
str(test)


# convert the label into factor for classification purpose
train$Made.Donation.in.March.2007 <- as.factor(train$Made.Donation.in.March.2007)

# split training data for cross validation
set.seed(410)
cv_train_rows <- sample(nrow(train), floor(nrow(train)*0.85))
cv_train <- train[cv_train_rows,]
cv_test <- test[-cv_train_rows,]

## Note: In real world, we need to do more feature engineering before the following operations!

