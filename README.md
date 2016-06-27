# Hanhan_Data_Science_Practice
data analysis, big data development, cloud, and any other cool things!


********************************************

* R Basics Data Analysis Practice
  * Problem Statement: http://datahack.analyticsvidhya.com/contest/practice-problem-bigmart-sales-prediction
  * data set: R_basics_train.csv, R_basics_test.csv
  * cleaned data sets: new_train.csv, nnew_test.csv
  * R code: R_Basics.R
  * Spark R code: https://github.com/hanhanwu/Hanhan_Data_Science_Practice/blob/master/R_Basics_SparkR.R
 
<b>Note:</b> The Spark R Notebook I am using is community editon, because R version maybe lower, many package in R Basics have not been supported.


* R data.table Basics
 * data.table is much faster than R data.frame
 * free tutorial: https://www.datacamp.com/courses/data-analysis-the-data-table-way
 * R code: https://github.com/hanhanwu/Hanhan_Data_Science_Practice/blob/master/R_data.table_basics.R


* Deal with Imbalanced Dataset in Classification
 * R code: https://github.com/hanhanwu/Hanhan_Data_Science_Practice/blob/master/deal_with_imbalanced_dataset.R
 

* Data Modeling with H2O and R data.table
 * R code: https://github.com/hanhanwu/Hanhan_Data_Science_Practice/blob/master/h2o_data.table.R
 * dataset: https://github.com/hanhanwu/Hanhan_Data_Science_Practice/blob/master/h2o_data.zip
 

* 7 Commonly used R data summary methods
 * R code: https://github.com/hanhanwu/Hanhan_Data_Science_Practice/blob/master/R_summarize_methods.R


* 5 R packages to help deal with missing values
 * R code: https://github.com/hanhanwu/Hanhan_Data_Science_Practice/blob/master/5R_packages_for_missing_values.R
 * My R code also contains how to compare the performace of missForest and Hmisc, this can be extended to other packages/algorithms, especailly about how to convert the imputed data into the format can be read by `mixError()` method in missForest.
 * <b>MICE</b> package creator description: http://www.stefvanbuuren.nl/mi/MI.html
 * <b>Amelia</b> package: This package (Amelia II) is named after Amelia Earhart, the first female aviator to fly solo across the Atlantic Ocean. History says, she got mysteriously disappeared (missing) while flying over the pacific ocean in 1937, hence this package was named to solve missing value problems. (The name of this package is, interesting)
 * <b>missForest</b> package: it is implemented on Random Forest, and offers methods to check imputation error, actual imputation accuracy, simply by tuning the parameters of the missForest() function, we can lower the error rate
 * <b>Hmisc</b> package: Hmisc automatically recognizes the variables types and uses bootstrap sample and predictive mean matching to impute missing values. You don’t need to separate or treat categorical variable. However, missForest can outperform Hmisc if the observed variables supplied contain sufficient information.
 * <b>mi</b> package: It allows graphical diagnostics of imputation models and convergence of imputation process. It uses bayesian version of regression models to handle issue of separation. Imputation model specification is similar to regression output in R. It automatically detects irregularities in data such as high collinearity among variables. Also, it adds noise to imputation process to solve the problem of additive constraints.
 * To sum up, when doing imputation, choose Hmisc and missForest first can be a good choice, followed by MICE


* Try Pandas
 * data input: https://github.com/hanhanwu/Hanhan_Data_Science_Practice/blob/master/GBM_Test.csv
 * python code: https://github.com/hanhanwu/Hanhan_Data_Science_Practice/blob/master/try_pandas.py
 * I really like pandas indexing and selection methods: http://pandas.pydata.org/pandas-docs/stable/indexing.html#indexing-label

<b>Note:</b> I'm using Spark Python Notebook, some features are unique there. Because my own machine could not install the right numpy version for pandas~


* Deal With Continuous Variables
 * Tutorial reference: http://www.analyticsvidhya.com/blog/2015/11/8-ways-deal-continuous-variables-predictive-modeling/?utm_content=buffer346f3&utm_medium=social&utm_source=facebook.com&utm_campaign=buffer
 * R code: https://github.com/hanhanwu/Hanhan_Data_Science_Practice/blob/master/deal_with_continuos_variables.R

********************************************

DIMENSION REDUCTION

* PCA (Principle Component Analysis) - R Version

 * R code: https://github.com/hanhanwu/Hanhan_Data_Science_Practice/blob/master/PCA_practice.R
 * data set: PCA_train.csv, PCA_test.csv
 * Why using One Hot encoding to convert categorical data into numerical data and only choose the top N columns after using PCA is right: http://stats.stackexchange.com/questions/209711/why-convert-categorical-data-into-numerical-using-one-hot-encoding

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


********************************************

KAGGLE PRACICE

* Who Will Purchase The Insurance

 * dataset: https://www.kaggle.com/c/homesite-quote-conversion/data
 * R code: https://github.com/hanhanwu/Hanhan_Data_Science_Practice/blob/master/who_will_purchase_insurance.R
 * Spark Python code: https://github.com/hanhanwu/Hanhan_Data_Science_Practice/blob/master/who_will_purchase_insurance.py
 * generate dataset for my Spark python code: https://github.com/hanhanwu/Hanhan_Data_Science_Practice/blob/master/insurance_spark_data.R
 
 -- Notes
 * Dimensional Reduction: I tried FAMD model first, since it supposed to handle the mix of categorical and numerical data. But my laptop didn't have enough memory to finish this. Then I changed to PCA, but needed to convert categorical data into numerical data myself first. After running PCA, it shows that the first 150-180 columns comtain the major info of the data.
 * About FAMD: FAMD is a principal component method dedicated to explore data with both continuous and categorical variables. It can be seen roughly as a mixed between PCA and MCA. More precisely, the continuous variables are scaled to unit variance and the categorical variables are transformed into a disjunctive data table (crisp coding) and then scaled using the specific scaling of MCA. This ensures to balance the influence of both continous and categorical variables in the analysis. It means that both variables are on a equal foot to determine the dimensions of variability. This method allows one to study the similarities between individuals taking into account mixed variables and to study the relationships between all the variables. It also provides graphical outputs such as the representation of the individuals, the correlation circle for the continuous variables and representations of the categories of the categorical variables, and also specific graphs to visulaize the associations between both type of variables. https://cran.r-project.org/web/packages/FactoMineR/FactoMineR.pdf
 * The predictive analysis part in R code is slow for SVM and NN by using my laptop (50GB disk memory availabe). Even though 150 features have been chosen from 228 features
 * Spark Python is much faster, but need to convert the .csv file data into LabeledPoint for training data, and SparseVector for testing data.
 * In my Spark Python code, I have tried SVM with SGD, Logistic Regression with SGD and Logistic Regression with LBFGS, but when I tune the parameters for SVM and Logistic Regression with SGD, they always returned an empty list wich should show those people who will buy insurance. Logistic Regression with LBFGS gives better results.


*********************************************

OTHER

* RDRToolbox - A R package for nonlinear dimensional reduction
 * How to install RDRToolbox: https://github.com/hanhanwu/Hanhan_Data_Science_Practice/blob/master/install_RDRToolbox.R
