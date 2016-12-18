# Method 1 - impute missing data with median/mode
q2 <- cbind(fact_data, num_data)
summarizeColumns(q2)
q2 <- mlr::impute(data.frame(q2), classes = list(factor = imputeMode(), integer = imputeMean()), dummy.classes = c("integer","factor"), dummy.type = "numeric")
q2 <- q2$data
summarizeColumns(q2)


# Method 2 - impute missing data with KNN, it will normalize data at the same time
q2 <- cbind(fact_data, num_data)
summarizeColumns(q2)
fact_cols <- lapply(q2, is.factor)
fc <- colnames(subset(q2, select = fact_cols == T))
q2[,(fc) := lapply(.SD, as.numeric), .SDcols = fc]
q2$HasWriteOff <- as.factor(q2$HasWriteOff)
preProcValues <- preProcess(q2, method = c("knnImpute","center","scale"))   # only deals with numerical data, normalized the data too
library('RANN')
q2_processed <- predict(preProcValues, q2)
summarizeColumns(q2_processed)


# remove 0 variance feature
zero_variance_list <- has_zero_variance(q2)
zero_variance_list


# remove highly correlation data
fact_cols <- sapply(q2, is.factor)
q2_num_data <- data.table(subset(q2, select = fact_cols==F))
q2_fact_data <- data.table(subset(q2, select = fact_cols==T))
ax <-findCorrelation(x = cor(q2_num_data), cutoff = 0.7)   # 0.7 is the threshold here
summarizeColumns(q2_num_data)
sort(ax)
q2_num_data <- q2_num_data[, -ax, with=F]
q2 <- cbind(q2_num_data, q2_fact_data)
rm(q2_num_data)
rm(q2_fact_data)


# Method 1 - deal with outliers with median/mode
boxplot(q2$LoanAmount)
q2$LoanAmount[which(q2$LoanAmount>12)] <- median(q2$LoanAmount)
boxplot(q2$LoanAmount)

# Method 2 - deal with outliers with binning
boxplot(q2$mlbf_InterestRate)
num_distribution_plot(q2$mlbf_InterestRate, q2)
num_distribution_plot(sqrt(q2$mlbf_InterestRate), q2)
quantile(q2$mlbf_InterestRate)
q2$mlbf_InterestRate <- as.factor(ifelse (q2$mlbf_InterestRate <= 0.05, "0~0.05", ifelse(q2$mlbf_InterestRate >0.07, ">0.07", "0.05~0.07")))
summary(q2$mlbf_InterestRate)
