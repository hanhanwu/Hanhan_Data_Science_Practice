# Hanhan_Data_Science_Practice
data analysis, big data development, cloud, and any other cool things!


********************************************

* R Basics Data Analysis Practice

  * Problem Statement: http://datahack.analyticsvidhya.com/contest/practice-problem-bigmart-sales-prediction
  * data set: R_basics_train.csv, R_basics_test.csv
  * cleaned data sets: new_train.csv, nnew_test.csv
  * R code: R_Basics.R


********************************************

DIMENSION REDUCTION

* PCA (Principle Component Analysis) - R Version

 * R code: https://github.com/hanhanwu/Hanhan_Data_Science_Practice/blob/master/PCA_practice.R
 * data set: R_basics_train.csv, R_basics_test.csv

* PCA (Principle Component Analysis) - Python Version

 * Python code: https://github.com/hanhanwu/Hanhan_Data_Science_Practice/blob/master/PCA_practice.py
 * data set: PCA_combi.csv
 
* PLS (Partial Least Squares) - R Version

 * R code: https://github.com/hanhanwu/Hanhan_Data_Science_Practice/blob/master/PLS_practice.R
 * data set: R_basics_train.csv, R_basics_test.csv

* NOTE
 * PCA is a unsupervised method while PLS is a supervised method. Therefore PLS is used when you want to find columns closely related to the label column. When there is just 1 label column, use R plsreg1(), when there are 1+ label columns, use R plsreg2()
 * In the code, when I was using PCA, it has funtion to help rescale values, when I was using PLS, I wrote data rescaling code, when compare the result with the code without rescaling, the result was worse. I am wondering whether this is related to the parameters for data rescaling, though I have tried several to make sure all the data are in [0,1] range.
 * When using PLS, one hot encoding (used in my PCA practice) to convert categorical to numerical is not a good choice, since no matter it is plsreg1 or plsreg2, both response variavles and indicators (identifiers and the label columns) need to be put into the method. Using one hot encoding on all these variables, will generate too many variables.

********************************************

CLOUD for DATA SCIENCE

* Databrics Cloud (Spark Cluster)
 
 * Spark Cluster FileSystem commands:
 https://github.com/hanhanwu/Hanhan_Data_Science_Practice/blob/master/Spark_Cluster_hdfs_commands.py
 * show matplotlib and ggplot: https://community.cloud.databricks.com/?o=4000185439389250#externalnotebook/https%3A%2F%2Fdocs.cloud.databricks.com%2Fdocs%2Flatest%2Fdatabricks_guide%2Findex.html%2304%2520Visualizations%2F4%2520Matplotlib%2520and%2520GGPlot.html


* Azure Machine Learning (AzureML)
  
 * AzureML Studio R Script with Time Series Forecasting: https://github.com/hanhanwu/Hanhan_Data_Science_Practice/blob/master/AzureML_time_series_forcasting.md
