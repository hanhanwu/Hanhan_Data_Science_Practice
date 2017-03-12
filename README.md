# Hanhan_Data_Science_Practice
data analysis, big data development, cloud, and any other cool things!


********************************************

<b> BIG DATA! </b> - Fantastic

* Why Spark is great?!
 * Spark is awesome to deal with big data problems! My godness, before I work on real big data, I just thought it was cool and smart! Today I have realized, it is super fantastic! Especially after I have written normal Python iterate code on 4000000 text records (2G) to extract multiple patterns....
 * My normal python iterate code, method 1: https://github.com/hanhanwu/Hanhan_Data_Science_Practice/blob/master/large_file_python_iterate_method1.py
 * My normal python iterate code, method 2: https://github.com/hanhanwu/Hanhan_Data_Science_Practice/blob/master/large_file_python_iterate_method2.py
 * My Spark Python code: https://github.com/hanhanwu/Hanhan_Data_Science_Practice/blob/master/large_file_Spark_Python.py
 * Running time comparison: Using the normal pyhton code, both methods (method 2 is a little bit faster), they took 20 minutes to run 5000 records. When I was running the Spark code to get insights of those 4000000 records, it took less than 10 minutes to give me what I want.
 * In Spark, remember to use cache() on the data you need to repeatedly use later. This will store the data on cache and reuse it without re-run all the previous steps. If you don't use cache(), it can be very slow. In my case, I'm cacheing `extracted_pattern_df`
 * <b>Note</b>: When you are running a big job through terminal command line, and want to stop the job before it finished execution, press `Control + C`, this is very helpful, trust me 
* How to run Spark through terminal command line
 * Download Spark here: https://spark.apache.org/downloads.html
 * Unpack that somewhere you like. Set an environment variable so you can find it easily later (CSH and BASH versions):


`setenv SPARK_HOME /home/you/spark-1.5.1-bin-hadoop2.6/`


`export SPARK_HOME=/home/you/spark-1.5.1-bin-hadoop2.6/`
 * Then `${SPARK_HOME}/bin/spark-submit --master local [your code file path] [your large data file path as input, this one only exist when you have sys.argv[1]]`


* Automation
 * Automation work is also necessary for Big Data, you can use PowerShell to creat .cmd file and schedule a task in your OS. For work related to database like SQL Server, Oracle, you can use their own scheduler to schedule jobs
 * Oracle Scueduler Official: http://docs.oracle.com/cd/E11882_01/server.112/e25494/scheduse.htm#ADMIN034
 * Oracle Scueduler StackOverflow: http://stackoverflow.com/questions/12212147/best-way-to-run-oracle-queries-periodically
 * Oracle Time Zones: https://docs.oracle.com/cd/B13866_04/webconf.904/b10877/timezone.htm


 * My experience about using PowerShell to automate Hadoop Hive query (HQL): That was a lot of pain. Hive is already more difficult to use than SQL/TSQL because it has less functions. Then, when you need to embed HQL in PowerShell, my godness, it made my life more difficult, especially when the data was gigantic and each time when you need to make a tiny change, all the testing time could force you to work overtime... After finishing that work, I have realized, how good our relational database is and how smart Spark SQL is!


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
 * modified pyhton code: https://github.com/hanhanwu/Hanhan_Data_Science_Practice/blob/master/try_pandas.ipynb
 * I really like pandas indexing and selection methods: http://pandas.pydata.org/pandas-docs/stable/indexing.html#indexing-label
 * While implementing my modified pyhton code, I gradually felt, R is still much better than pandas in binning and ploting. R cna do those data query related work such as data merging as well. But I still like Pivot Table, Multi-Indexing, Corss Tab in pandas.
 * Reference: https://www.analyticsvidhya.com/blog/2016/01/12-pandas-techniques-python-data-manipulation/?utm_content=buffer34860&utm_medium=social&utm_source=facebook.com&utm_campaign=buffer

<b>Note:</b> I'm using Spark Python Notebook, some features are unique there. Because my own machine could not install the right numpy version for pandas~


* Deal With Continuous Variables
 * Tutorial reference: http://www.analyticsvidhya.com/blog/2015/11/8-ways-deal-continuous-variables-predictive-modeling/?utm_content=buffer346f3&utm_medium=social&utm_source=facebook.com&utm_campaign=buffer
 * R code: https://github.com/hanhanwu/Hanhan_Data_Science_Practice/blob/master/deal_with_continuos_variables.R


* Minimize Logloss
 * R code: https://github.com/hanhanwu/Hanhan_Data_Science_Practice/blob/master/minimize_logloss.R
 * Training data: https://github.com/hanhanwu/Hanhan_Data_Science_Practice/blob/master/blood_train.csv
 * Testing data: https://github.com/hanhanwu/Hanhan_Data_Science_Practice/blob/master/blood_test.csv
 * The 2 methods used in the code are: Platt Scaling and Isotonic Regression
 * Plotting Logloss can caputure changes that may not be able to be caputered by accuracy, AUC and other metrics
 * reference: http://www.analyticsvidhya.com/blog/2016/07/platt-scaling-isotonic-regression-minimize-logloss-error/?utm_content=buffer2f3d5&utm_medium=social&utm_source=facebook.com&utm_campaign=buffer


* Time Series Modeling
 * Time Series Prediction R code: https://github.com/hanhanwu/Hanhan_Data_Science_Practice/blob/master/time_series_modeling.R
 * Tutorial Reference: http://www.analyticsvidhya.com/blog/2015/12/complete-tutorial-time-series-modeling/?utm_content=buffer529c5&utm_medium=social&utm_source=facebook.com&utm_campaign=buffer


* GBM (Gradient Boostin) parma tuning
 * training data: https://github.com/hanhanwu/Hanhan_Data_Science_Practice/blob/master/GBM_Train.csv
 * testing data: https://github.com/hanhanwu/Hanhan_Data_Science_Practice/blob/master/GBM_Test.csv
 * data preprocessing: https://github.com/hanhanwu/Hanhan_Data_Science_Practice/blob/master/GBM_data_preprocessing.R
 * cleaned training data: https://github.com/hanhanwu/Hanhan_Data_Science_Practice/blob/master/GBM_new_train.csv
 * cleaned testing data: https://github.com/hanhanwu/Hanhan_Data_Science_Practice/blob/master/GBM_new_test.csv
 * GBM param tuning: https://github.com/hanhanwu/Hanhan_Data_Science_Practice/blob/master/GBM_param_tuning.ipynb
 * This is one of the most challenge practice I have done, not only because the way to preprocess the data is different from what I have learned before, but also because of the way it tune params made my machine pretty hot and sacred me (I love my machine and really make it function well for a very long time). <b>But!</b> Have to admit that this practice is also one of the most useful one! Not only because of its preprocessing method is useful, but also because the way it tune params and the order of tuning params are very practical. I was always wondering which params to tune and should be in which order. Now the methods used here have answered all my previous questions.
 * Reference: http://www.analyticsvidhya.com/blog/2016/02/complete-guide-parameter-tuning-gradient-boosting-gbm-python/?utm_content=bufferc754b&utm_medium=social&utm_source=facebook.com&utm_campaign=buffer
 

* R MLR package
 * This packages contains algorithms and methods for data preprocessing and machine learning, ALL IN ONE
 * Its impute method is good to impute all the columns have NAs, but didn't work well when I tried to impute individual columns. Because its impute will generate dummy columns for columns with NAs, using this methods needs both training and testing data has the same columns that contain NAs. But if they don't, we can still use this impute method but need to remove dummy columns only in training data or testing data, this is fine because the original columns have been imputed at the same time.
 * R code: https://github.com/hanhanwu/Hanhan_Data_Science_Practice/blob/master/MLR_basics.R
 * training data: https://github.com/hanhanwu/Hanhan_Data_Science_Practice/blob/master/MLR_train.csv
 * testing data: https://github.com/hanhanwu/Hanhan_Data_Science_Practice/blob/master/MLR_test.csv
 * Reference: https://www.analyticsvidhya.com/blog/2016/08/practicing-machine-learning-techniques-in-r-with-mlr-package/?utm_source=feedburner&utm_medium=email&utm_campaign=Feed%3A+AnalyticsVidhya+%28Analytics+Vidhya%29


* DIY Ensembling
 * reference: https://www.analyticsvidhya.com/blog/2017/02/introduction-to-ensembling-along-with-implementation-in-r/?utm_source=feedburner&utm_medium=email&utm_campaign=Feed%3A+AnalyticsVidhya+%28Analytics+Vidhya%29
 * My code: https://github.com/hanhanwu/Hanhan_Data_Science_Practice/blob/master/DIY_ensembling.R
 * data: https://github.com/hanhanwu/Hanhan_Data_Science_Practice/blob/master/DIY_ensembling_loan_data.csv
 * One important thing I have learned is, when each model provides highly correlated prediction results, ensembling them together may not give better results. So, <b> check the correlation of their prediction results first,</b> then decide whether you want to ensembling them
 

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


* Easy simple way to do feature selection with Boruta (so far it's the most convenient feature selection method I have tried):
 * R code: https://github.com/hanhanwu/Hanhan_Data_Science_Practice/blob/master/feature_selection_Boruta.R
 * Training data: https://github.com/hanhanwu/Hanhan_Data_Science_Practice/blob/master/loan_prediction_train.csv
 * Testing data: https://github.com/hanhanwu/Hanhan_Data_Science_Practice/blob/master/loan_prediction_test.csv
 * Tutorial: http://www.analyticsvidhya.com/blog/2016/03/select-important-variables-boruta-package/?utm_content=bufferec6a6&utm_medium=social&utm_source=facebook.com&utm_campaign=buffer


********************************************

DATA PREPROCESSING

* Python scikit-learn preprocessing, with pandas
 * IPython Notebook: https://github.com/hanhanwu/Hanhan_Data_Science_Practice/blob/master/scikit_learn_preprocessing.ipynb
 * Tutorial reference: http://www.analyticsvidhya.com/blog/2016/07/practical-guide-data-preprocessing-python-scikit-learn/?utm_content=buffera1e2c&utm_medium=social&utm_source=facebook.com&utm_campaign=buffer
 * X_train: https://github.com/hanhanwu/Hanhan_Data_Science_Practice/blob/master/X_train.csv
 * Y_train: https://github.com/hanhanwu/Hanhan_Data_Science_Practice/blob/master/Y_train.csv
 * X_test: https://github.com/hanhanwu/Hanhan_Data_Science_Practice/blob/master/X_test.csv
 * Y_test: https://github.com/hanhanwu/Hanhan_Data_Science_Practice/blob/master/Y_test.csv

* Resolve Skewness
 * Many models assume data predictors are normaly distributed, which is evenly skewness. But in practice, this is almost impossible. Therefore, by checking the skewness of continuous factors and try to make them to get close to normal distribution is ncessary, for those models that have this accumption.
 * scipy.stats.boxcox: http://scipy.github.io/devdocs/generated/scipy.stats.boxcox.html
 * data for practice: https://github.com/hanhanwu/Hanhan_Data_Science_Practice/blob/master/168866123_T_ONTIME.csv
 * Original data source for download: http://www.transtats.bts.gov/DL_SelectFields.asp?Table_ID=236
 * Python Practice Code: https://github.com/hanhanwu/Hanhan_Data_Science_Practice/blob/master/resolve_skewness.ipynb
 
* Deal with Imbalanced Dataset
 * practice 1: https://github.com/hanhanwu/Hanhan_Data_Science_Practice/blob/master/deal_with_imbalanced_dataset.R
 * practice 2: https://github.com/hanhanwu/Hanhan_Data_Science_Practice/blob/master/deal_with_imbalanced_data_2.R
 * training data for practice 2: https://www.analyticsvidhya.com/wp-content/uploads/2016/09/train.zip
 * testing data for practice 2: https://www.analyticsvidhya.com/wp-content/uploads/2016/09/test.zip
 * data description for practice 2: http://archive.ics.uci.edu/ml/machine-learning-databases/census-income-mld/census-income.names
 * Tutorial: https://www.analyticsvidhya.com/blog/2016/09/this-machine-learning-project-on-imbalanced-data-can-add-value-to-your-resume/?utm_source=feedburner&utm_medium=email&utm_campaign=Feed%3A+AnalyticsVidhya+%28Analytics+Vidhya%29

* TOOL - Google OpenRefine
 * [tutorial 1][1] - text facet and clustering to help organize inconsistent text data (eg. "White and Black", "white and black" are the same); numerical facet has interactive interface to allow you check outliers, calculate log and errors.
 * Practice data (registered csv): http://data.vancouver.ca/datacatalogue/animalInventory.htm
 * [Clustering algorithm that helps group similar text][2]
 * Note: compared the clustering algorithms used in Google OpenRefine, Levenshtein Distance match strings is is order sensitive, which means similar words should appear in the same order so that the distance between 2 strings could be shorter.
 * [tutorial 2][3] - data transformation, convert data into table and transform the data in table (such as sperate columns, format and generate new columns, etc.); export with DIY template; copy all the redo list to the new similar data file and finish automatic data transformation. It is using wiki text data as the example
 * [tutorial 3][4] - data enrich, fetch urls for the data by using web services such as those Google web services to generate new columns; add new data using Freebase Reconcilation Service to add more data for each row
 
 
 
 [1]:https://www.youtube.com/watch?v=B70J_H_zAWM
 [2]:http://www.padjo.org/tutorials/open-refine/clustering/#clustering-algorithms
 [3]:https://www.youtube.com/watch?v=cO8NVCs_Ba0
 [4]:https://www.youtube.com/watch?v=5tsyz3ibYzk


********************************************

TREE BASED MODELS

* Tree based models in detail with R & Python example: https://www.analyticsvidhya.com/blog/2016/04/complete-tutorial-tree-based-modeling-scratch-in-python/?utm_content=bufferade26&utm_medium=social&utm_source=facebook.com&utm_campaign=buffer


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

