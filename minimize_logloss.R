
## Reference: http://www.analyticsvidhya.com/blog/2016/07/platt-scaling-isotonic-regression-minimize-logloss-error/?utm_content=buffer2f3d5&utm_medium=social&utm_source=facebook.com&utm_campaign=buffer

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

cv_predict_platt <- predict(model_LogisticRegresion, df[1], type = "response")
LogLoss(as.numeric(as.character(cv_test$Made.Donation.in.March.2007)), cv_predict_platt)
## 3.575357

# reality plot
plot(c(0,1),c(0,1), col="grey",type="l",xlab = "Mean Prediction",ylab="Observed Fraction")
reliability.plot <- function(obs, pred, bins=10, scale=T) {
  # Plots a reliability chart and histogram of a set of predicitons from a classifier
  #
  # Args:
  # obs: Vector of true labels. Should be binary (0 or 1)
  # pred: Vector of predictions of each observation from the classifier. Should be real
  # number
  # bins: The number of bins to use in the reliability plot
  # scale: Scale the pred to be between 0 and 1 before creating reliability plot
  require(plyr)
  library(Hmisc)
  min.pred <- min(pred)
  max.pred <- max(pred)
  min.max.diff <- max.pred - min.pred
  if (scale) {
    pred <- (pred - min.pred) / min.max.diff 
  }
  bin.pred <- cut(pred, bins)
  k <- ldply(levels(bin.pred), function(x) {
    idx <- x == bin.pred
    c(sum(obs[idx]) / length(obs[idx]), mean(pred[idx]))
  })
  not.nan.idx <- !is.nan(k$V2)
  k <- k[not.nan.idx,]
  return(k)
}


## The one closer to the grey line is more accurate
# Without Platt Scaling
k1 <-reliability.plot(as.numeric(as.character(cv_train$Made.Donation.in.March.2007)),cv_predict$`1`,bins = 5)
k1
lines(k1$V2, k1$V1, xlim=c(0,1), ylim=c(0,1), xlab="Mean Prediction", ylab="Observed Fraction", col="red", type="o", main="Reliability Plot")
# With Platt Scaling
k2 <-reliability.plot(as.numeric(as.character(cv_train$Made.Donation.in.March.2007)),cv_predict_platt,bins = 5)
k2
lines(k2$V2, k2$V1, xlim=c(0,1), ylim=c(0,1), xlab="Mean Prediction", ylab="Observed Fraction", col="blue", type="o", main="Reliability Plot")

legend("topright",lty=c(1,1),lwd=c(2.5,2.5),col=c("blue","red"),legend = c("platt scaling","without plat scaling"))



# Do Prediction on Test data

## without platt scaling, probability prediction for both label values (0,1)
predict_test <- as.data.frame(predict(model_rf, newdata = test, type = "prob"))
summary(predict_test)
str(predict_test)

## with platt scaling, probability prediction for label value 1
df_test <- data.frame(predict_test$`1`)
str(df_test)
colnames(df_test) <- c("predict_value")
predict_test_platt <- predict(model_LogisticRegresion, df_test, type = "response")
summary(predict_test_platt)
str(predict_test_platt)

