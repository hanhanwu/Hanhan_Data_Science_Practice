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

# check all the classification learners you can use in mlr
listLearners("classif", check.packages = F)[c("class","package")]

# connect to the SQL Server database
q <- odbcDriverConnect('driver={SQL Server};server=[your SQL Server server];database=[your database];trusted_connection=true')
q1 <- sqlQuery(q, "[your query]")
setDT(q1)
summarizeColumns(q1)

# remove IDs
q1[, [your_ID_name]:= NULL]

summarizeColumns(q1)

# make a copy of the data.table as data.frame
## this is because the copy of data.table will be a reference, which shares the same memory location, when you make change on
## 1 data.table, all changed
df_q1 <- data.frame(q1)
q1_copy <- df_q1
rm(df_q1)
# to check data.table/data.frame address
address(q1)
address(q1_copy)
