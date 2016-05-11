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
