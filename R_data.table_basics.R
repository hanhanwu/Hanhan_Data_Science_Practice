path<- "[your workspace path]"
setwd(path)

library(data.table)

# Create a data table
dt1 <- data.table(x = c("a", "b", "c", "d", "e"), y = c(1, 2, 3, 4, 5))

# Create a data.table using recycling
dt2 <- data.table(a = c(1L, 2L), b = LETTERS[1:4])

# Print the third row to the console
dt2[3,]

# Print the second and third row to the console, but do not commas
dt2[2:3,]


# Print the penultimate row of DT, .N indicates the total rows
dt2[.N - 1]

# Print the column names of DT, and number of rows and number of columns
colnames(dt2)
dim(dt2)

# Select row 2 twice and row 3, returning a data.table with three rows 
## where row 2 is a duplicate of row 1.
dt2[c(2,2,3)]


# About Column j
## create a data table
DT <- data.table(a=c(1,2,3,4,5), b=LETTERS[1:5], c=c(6,7,8,9,10))
## select a column and output as a data.table
DT[,.(b)]
## select a column and output it as a vector
DT[,(b)]
## Note: when using .() in j position, the output is always a data.table


## non-existing column
D <- 5
DT[, .(D)]  # output 5 as a data.table
DT[, (D)]  # output 5 as vector

## subsetting data.table
subDT <- DT[1:3, .(b,c)]  # only get column b,c with the first 3 rows
ans <- DT[, .(b, val=a*c)]  # val is the product of column a, c
target <- data.table(b = c("a", "b", "c", "d", "e", "a", "b", "c", "d", "e"), 
                     val = as.integer(c(6:10, 1:5)))
# ans2 is the same as target
ans2 <- DT[, .(b, val=as.integer(c(6:10, 1:5)))]


# Column by
DT <- as.data.table(iris)
## For each Species, print the mean Sepal.Length
DT[, mean(Sepal.Length), by=Species]
## Print mean Sepal.Length, grouping by first letter of Species
DT[, mean(Sepal.Length), by=substr(Species, 1, 1)]
## order the columns, "-" means descending order
order_DT <- DT[order(Sepal.Length, -Sepal.Width)]
head(order_DT)
## add a column
DT[, new_col:= Sepal.Width+Sepal.Length]
head(DT)
## update a column
table(DT$Species)
DT[Species == 'setosa', Species:= 'Setosa']
table(DT$Species)
## delete columns
head(DT)
DT[, c("new_col") := NULL]
head(DT)
## chaining of changes
DT[, new_col:= Sepal.Width+Sepal.Length][Species == 'setosa', Species:= 'Setosa'][, c("new_col") := NULL]
## use keys for subset
setkey(DT, Species)
sub_DT1 <- DT[.("Setosa")]
sub_DT2 <- DT[.("Setosa", "virginica")]   # the second class has been added as a new column
dim(sub_DT1)
dim(sub_DT2)


## use .N in j with by, will count number of rows in each group

## Group the specimens by Sepal area (to the nearest 10 cm2) 
## and count how many occur in each group
DT[, .(Count = .N), by=.(Area = 10*round(Sepal.Length*Sepal.Width/10))]

## calculate cumulative sum of C in column C while you group by A,B. 
## Store it in a new data.table DT2 with 3 columns: A, B and C
set.seed(1L)
DT <- data.table(A = rep(letters[2:1], each = 4L), 
                 B = rep(1:4, each = 2L), 
                 C = sample(8))
DT2 <- DT[, .(C = cumsum(C)), by = .(A, B)]
## # Select from DT2 the last two values from C while you group by A
DT2[, .(C = tail(C, 2)), by = A]
