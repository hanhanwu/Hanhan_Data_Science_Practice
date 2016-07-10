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
cv_test <- train[-cv_train_rows,]

## Note: In real world, we need to do more feature engineering before the following operations!

library(randomForest)
model_rf <- randomForest(Made.Donation.in.March.2007~., data = cv_train, keep.forest = T, importance = T)
cv_predict <- as.data.frame(predict(model_rf, type = "prob"))
str(cv_predict)

# predefine LogLoss function
LogLoss<-function(act, pred)
{
  eps = 1e-15;
  nr = length(pred)
  pred = matrix(sapply( pred, function(x) max(eps,x)), nrow = nr) 
  pred = matrix(sapply( pred, function(x) min(1-eps,x)), nrow = nr)
  ll = sum(act*log(pred) + (1-act)*log(1-pred))
  ll = ll * -1/(length(act)) 
  return(ll);
}

# calculate logloss wihtout Platt Scaling
LogLoss(as.numeric(as.character(cv_test$Made.Donation.in.March.2007)), cv_predict$`1`)
## 11.9268

# using Platt Scaling
df <- data.frame(cv_predict$`1`, cv_train$Made.Donation.in.March.2007)
colnames(df) <- c("predict_value", "actual_value")
model_LogisticRegresion <- glm(actual_value~predict_value, data = df, family = binomial)
