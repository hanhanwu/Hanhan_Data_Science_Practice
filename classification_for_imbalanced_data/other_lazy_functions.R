# I'd rather spend a little more time to write functions that can be executed in a batch, than checking features one by one


# This function is to findhow much percentage of data points drop out of [avg-sd, avg+sd] range
# It does this for each feature in both data sets, and compare the percentages, as well as percentage differences
range_diff <- function(my_df1, my_df2, col_lst){
  result_vector <- c()
  
  for (col_str in col_lst)
  {
    out1 <- my_df1[[col_str]][which(my_df1[[col_str]] > my_df1[[paste(col_str, "right1", sep="_")]] | my_df1[[col_str]] < my_df1[[paste(col_str, "left1", sep="_")]])]
    out_perct1 <- length(out1)/length(my_df1[[col_str]]) # my_df1[[col_str]] is calling the column through its string name
    
    out2 <- my_df2[[col_str]][which(my_df2[[col_str]] > my_df2[[paste(col_str, "right1", sep="_")]] | my_df2[[col_str]] < my_df2[[paste(col_str, "left1", sep="_")]])]
    out_perct2 <- length(out2)/length(my_df2[[col_str]])
    
    out_diff <- out_perct1 - out_perct2
    
    result_vector <- c(result_vector, paste(col_str, paste(paste(out_perct1, out_perct2, sep = " "), out_diff, sep = " "), sep = " "))
  }
  result_df <- read.table(text = result_vector)
  colnames(result_df) <- c("features", "out_perct1", "out_perct2", "out_perct_diff")
  return(result_df)
}

## Generate avg&sd range for each accountid
df_avg <- aggregate(df[, 11:32], list(df$name), mean)  # calculate avg group by name
head(df_avg)
dim(df_avg)

df_sd <- aggregate(df[, 11:32], list(df$name), sd)  # calculate standard deviation group by name
head(df_sd)
dim(df_sd)

num_cols <- colnames(df_avg[,2:23])

df_avg_plus_sd <- df_avg[num_cols] + df_sd[num_cols]
head(df_avg_plus_sd)
colnames(df_avg_plus_sd) <- paste(colnames(df_avg_plus_sd), "right1", sep="_")
head(df_avg_plus_sd)

df_avg_minus_sd <- df_avg[num_cols] - df_sd[num_cols]
head(df_avg_minus_sd)
colnames(df_avg_minus_sd) <- paste(colnames(df_avg_minus_sd), "left1", sep="_")
head(df_avg_minus_sd)

df1 <- cbind(df_avg_minus_sd, df_avg_plus_sd)
df1[,'name'] <- df_avg$Group.1

original_cols <- df[, c(2,11:32)]
merged_df1 <- merge(x=original_cols, y=df1, by = "name", all.x = T)  # join by name
summarizeColumns(merged_df1)

# do the same thing to get merged_df2, then
range_diff_df <- range_diff(merged_df1, merged_df2, num_cols)
with(range_diff_df, range_diff_df[order(-out_perct_diff), ])  # order by percentage difference in descending order

################################################################################################################
