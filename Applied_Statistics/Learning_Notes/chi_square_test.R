# -------------------------------------- #
## Chi-Square Test
# -------------------------------------- #
library(data.table)

path<- "/Users/hanhanwu/Desktop/R/"
setwd(path)

df1 <- fread('chi1.csv')
head(df1)
dim(df1)

# print out Proportion Table for Observed Frequencies
prop.table(table(df1$`Experience intervals`))
# 11 - 20 Years 21 - 40 Years  6 - 10 Years  Upto 5 Years 
# 0.2312925     0.1408163     0.4129252     0.2149660 

# calculate chi-square with observed & expected frequencies
chisq.test(x = table(df1$`Experience intervals`),  # observed 
           p = c(0.2, 0.17, 0.41, 0.22))           # expected
# X-squared = 14.762, df = 3, p-value = 0.002032
## Here, since p-value is smaller than 0.05, reject Null Hypothesis, that is to say,
## observed frequencies and expected frequencies are significant different from each other.


# -------------------------------------- #
## Chi-Square Test for Association/Independence

## Here to check whether Age and Experience are independent
# -------------------------------------- #
ct <- table(df1$`age intervals`, df1$`Experience intervals`)
ct
# 11 - 20 Years 21 - 40 Years 6 - 10 Years Upto 5 Years
# 18 - 30            22             0          172          192
# 31 - 40           190            20          308          101
# 41 - 50            85           112          110           15
# 51 - 60            43            75           17            8

chisq.test(ct)
# X-squared = 679.97, df = 9, p-value < 2.2e-16
## p-value is very small, reject Null Hypothesis, so the 2 variables are dependent
