# I'd rather spend a little more time to write functions that can be executed in a batch, than checking features one by one


# This function is to findhow much percentage of data points drop out of [avg-sd, avg+sd] range
# It does this for each feature in both data sets, and compare the percentages, as well as percentage differences
range_diff <- function(my_df1, my_df2, col_lst){
  result_vector <- c()
  
  for (col_str in col_lst)
  {
    out1 <- my_df1[[col_str]][which(my_df1[[col_str]] > my_df1[[paste(col_str, "right1", sep="_")]] | my_df1[[col_str]] < my_df1[[paste(col_str, "left1", sep="_")]])]
    out_perct1 <- length(out1)/length(my_df1[[col_str]])
    
    out2 <- my_df2[[col_str]][which(my_df2[[col_str]] > my_df2[[paste(col_str, "right1", sep="_")]] | my_df2[[col_str]] < my_df2[[paste(col_str, "left1", sep="_")]])]
    out_perct2 <- length(out2)/length(my_df2[[col_str]])
    
    out_diff <- out_perct1 - out_perct2
    
    result_vector <- c(result_vector, paste(col_str, paste(paste(out_perct1, out_perct2, sep = " "), out_diff, sep = " "), sep = " "))
  }
  result_df <- read.table(text = result_vector)
  colnames(result_df) <- c("features", "out_perct1", "out_perct2", "out_perct_diff")
  return(result_df)
}
