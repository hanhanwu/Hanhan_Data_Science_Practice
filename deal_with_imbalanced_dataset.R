library(ROSE)

# built in dataset
data(hacide)

# quick explore data
str(hacide.train)
table(hacide.train$cls)     # number of each class
prop.table(table(hacide.train$cls))    # class distribution

# Experiment: see how badly imbalanced dataset will influence accuracy first
library(rpart)
imb_tree <- rpart(cls~., data=hacide.train)
imb_pred <- predict(imb_tree, newdata = hacide.test)
## test accuracy
accuracy.meas(hacide.test$cls, imb_pred[,2])
## precision = 1 means there is no false positive
## low recall means there are higher number of false negative
## low F score indicates low accuracy
roc.curve(hacide.test$cls, imb_pred[,2], plotit = T)
## AUC = 0.6 is a low score

# Wihtout dealing with imbalancing problem, 
# the algorithm gets biased toward the majoriy class


# SAMPLING

# over sampling, may lead to overfitting
bal_data_over <- ovun.sample(cls~., data = hacide.train, method = "over", N = 1960, seed = 410)$data
table(bal_data_over$cls)

# under sampling, may lose important data
bal_data_under <- ovun.sample(cls~., data = hacide.train, method = "under", N = 40, seed = 410)$data
table(bal_data_under$cls)

# do both over-sampling and under-smapling
## p means the probability of positive classes in the new dataset
bal_data_both <- ovun.sample(cls~., data = hacide.train, method = "both", p = 0.5, N = 1000, seed = 410)$data
table(bal_data_both$cls)

# ROSE helps overcome the shortcomings in over-smapling and under-sampling
bal_data_rose <- ROSE(cls~., data = hacide.train, seed = 410)$data
table(bal_data_rose$cls)


# decision trees for each data
tree_rose <- rpart(cls~., data = bal_data_rose)
tree_both <- rpart(cls~., data = bal_data_both)
tree_over <- rpart(cls~., data = bal_data_over)
tree_under <- rpart(cls~., data = bal_data_under)

pred_rose <- predict(tree_rose, newdata = hacide.test)
pred_both <- predict(tree_both, newdata = hacide.test)
pred_over <- predict(tree_over, newdata = hacide.test)
pred_under <- predict(tree_under, newdata = hacide.test)

roc.curve(hacide.test$cls, pred_rose[,2])
roc.curve(hacide.test$cls, pred_both[,2])
roc.curve(hacide.test$cls, pred_over[,2])
roc.curve(hacide.test$cls, pred_under[,2])

# by using ROSE, we can also use "holdout"
# to prevent high variance
ROSE_holdout <- ROSE.eval(cls ~ ., data = hacide.train, 
                          learner = rpart, method.assess = "holdout", 
                          extr.pred = function(obj)obj[,2], seed = 410)


# plot all the ROC curve together
library(ROCR)
pd1 <- prediction(pred_rose[,2], hacide.test$cls)
pf1 <- performance(pd1, "tpr","fpr")

pd2 <- prediction(pred_both[,2], hacide.test$cls)
pf2 <- performance(pd2, "tpr","fpr")

pd3 <- prediction(pred_over[,2], hacide.test$cls)
pf3 <- performance(pd3, "tpr","fpr")

pd4 <- prediction(pred_under[,2], hacide.test$cls)
pf4 <- performance(pd4, "tpr","fpr")

plot(pf1, colorize = TRUE)
plot(pf2, add = TRUE, colorize = TRUE)
plot(pf3, add = TRUE, colorize = TRUE)
plot(pf4, add = TRUE, colorize = TRUE)


# An alternative easier way to plot all the ROC curves together with AUC

library(pROC)
p<- plot(roc(hacide.test$cls, pred_rose[,2]), print.auc = TRUE, col = "blue")
p <- plot(roc(hacide.test$cls, pred_both[,2]), print.auc = TRUE, 
                       col = "green", print.auc.y = .4, add = TRUE)
