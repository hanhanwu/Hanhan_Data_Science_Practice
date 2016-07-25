path<- "[your file path]"
setwd(path)

library(data.table)

# using stringAsFactor so that categorical data will be easier to read in summary()
train <- fread("GBM_train.csv", stringsAsFactors = T)
test <- fread("GBM_test.csv", stringsAsFactors = T)

dim(train)
dim(test)

summary(train)
str(train)
colSums(is.na(train))

# get number of unique values, 698 different cities in this case
dim(unique(train[, .(City)]))
# remove City since there are too many values
train[, c("City") := NULL]
test[, c("City") := NULL]

# convert DOB to age
## convert to R standard Date format first
## If simply use as.Date(), some years wil be convert to the wrong format, 
## I have to use regex here
train$DOB <- as.character(train$DOB)
str(train$DOB)
ptn <- '(\\d\\d-\\w\\w\\w-)(\\d\\d)'
train$DOB <- sub(ptn, '\\119\\2', train$DOB)
str(train$DOB)

test$DOB <- as.character(test$DOB)
test$DOB <- sub(ptn, '\\119\\2', test$DOB)
str(test$DOB)

## convert DOB to Age, default is Age in months
library(eeptools)
train$DOB <- as.Date(train$DOB, "%d-%b-%Y")
str(train$DOB)
train$DOB <- floor(age_calc(train$DOB, units = "years"))   # you may get warning, it's ok
str(train$DOB)
summary(train$DOB)

test$DOB <- as.Date(test$DOB, "%d-%b-%Y")
test$DOB <- floor(age_calc(test$DOB, units = "years")) 
summary(test$DOB)

## rename DOB as Age
train[, Age := DOB]
summary(train)
train[, DOB := NULL]
summary(train)

test[, Age := DOB]
test[, DOB := NULL]
summary(test)

# drop EmployerName, which has many distinct values
train[, Employer_Name := NULL]
train[, ID := NULL]
train[, Lead_Creation_Date := NULL]
train[, LoggedIn := NULL]
train[, Salary_Account := NULL]
summary(train)

test[, Employer_Name := NULL]
test[, ID := NULL]
test[, Lead_Creation_Date := NULL]
test[, LoggedIn := NULL]
test[, Salary_Account := NULL]
summary(train)


train$EMI_Loan_Submitted <- ifelse(is.na(train$EMI_Loan_Submitted),1,0)
train[, EMI_Loan_Submitted_Missing := EMI_Loan_Submitted]
test$EMI_Loan_Submitted <- ifelse(is.na(test$EMI_Loan_Submitted),1,0)
test[, EMI_Loan_Submitted_Missing := EMI_Loan_Submitted]

train$Interest_Rate <- ifelse(is.na(train$Interest_Rate),1,0)
train[, Interest_Rate_Missing := Interest_Rate]
test$Interest_Rate <- ifelse(is.na(test$Interest_Rate),1,0)
test[, Interest_Rate_Missing := Interest_Rate]

train$Loan_Amount_Submitted <- ifelse(is.na(train$Loan_Amount_Submitted),1,0)
train[, Loan_Amount_Submitted_Missing := Loan_Amount_Submitted]
test$Loan_Amount_Submitted <- ifelse(is.na(test$Loan_Amount_Submitted),1,0)
test[, Loan_Amount_Submitted_Missing := Loan_Amount_Submitted]

train$Loan_Tenure_Submitted <- ifelse(is.na(train$Loan_Tenure_Submitted),1,0)
train[, Loan_Tenure_Submitted_Missing := Loan_Tenure_Submitted]
test$Loan_Tenure_Submitted <- ifelse(is.na(test$Loan_Tenure_Submitted),1,0)
test[, Loan_Tenure_Submitted_Missing := Loan_Tenure_Submitted]

train$Processing_Fee <- ifelse(is.na(train$Processing_Fee),1,0)
train[, Processing_Fee_Missing := Processing_Fee]
test$Processing_Fee <- ifelse(is.na(test$Processing_Fee),1,0)
test[, Processing_Fee_Missing := Processing_Fee]

train$Existing_EMI[is.na(train$Existing_EMI)] <- median(train$Existing_EMI, na.rm = TRUE)
test$Existing_EMI[is.na(test$Existing_EMI)] <- median(test$Existing_EMI, na.rm = TRUE)

train$Loan_Amount_Applied[is.na(train$Loan_Amount_Applied)] <- median(train$Loan_Amount_Applied, na.rm = TRUE)
test$Loan_Amount_Applied[is.na(test$Loan_Amount_Applied)] <- median(test$Loan_Amount_Applied, na.rm = TRUE)

train$Loan_Tenure_Applied[is.na(train$Loan_Tenure_Applied)] <- median(train$Loan_Tenure_Applied, na.rm = TRUE)
test$Loan_Tenure_Applied[is.na(test$Loan_Tenure_Applied)] <- median(test$Loan_Tenure_Applied, na.rm = TRUE)


# deal with missing values
colSums(is.na(train))
summary(train)
colSums(is.na(test))
summary(test)


write.csv(train, "GBM_new_train.csv")
write.csv(test, "GBM_new_test.csv")
