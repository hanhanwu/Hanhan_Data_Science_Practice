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
  * Unpack that somewhere you like. Set an environment variable so you can find it easily later (CSH and BASH versions): `setenv SPARK_HOME /home/you/spark-1.5.1-bin-hadoop2.6/`, `export SPARK_HOME=/home/you/spark-1.5.1-bin-hadoop2.6/`
  * Then `${SPARK_HOME}/bin/spark-submit --master local [your code file path] [your large data file path as input, this one only exist when you have sys.argv[1]]`


* Automation
  * Automation work is also necessary for Big Data, you can use PowerShell to creat .cmd file and schedule a task in your OS. For work related to database like SQL Server, Oracle, you can use their own scheduler to schedule jobs
  * Oracle Scueduler Official: http://docs.oracle.com/cd/E11882_01/server.112/e25494/scheduse.htm#ADMIN034
  * Oracle Scueduler StackOverflow: http://stackoverflow.com/questions/12212147/best-way-to-run-oracle-queries-periodically
  * Oracle Time Zones: https://docs.oracle.com/cd/B13866_04/webconf.904/b10877/timezone.htm
  * My experience about using PowerShell to automate Hadoop Hive query (HQL): That was a lot of pain. Hive is already more difficult to use than SQL/TSQL because it has less functions. Then, when you need to embed HQL in PowerShell, my godness, it made my life more difficult, especially when the data was gigantic and each time when you need to make a tiny change, all the testing time could force you to work overtime... After finishing that work, I have realized, how good our relational database is and how smart Spark SQL is!


********************************************

R PRACTICE

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
 
 
* Control Charts in R
  * reference: https://cran.r-project.org/web/packages/qicharts/vignettes/controlcharts.html
  * My code: https://github.com/hanhanwu/Hanhan_Data_Science_Practice/blob/master/control_charts.R
  

* Create Animated Visualization with R
  * reference: https://www.analyticsvidhya.com/blog/2017/06/a-study-on-global-seismic-activity-between-1965-and-2016/?utm_source=feedburner&utm_medium=email&utm_campaign=Feed%3A+AnalyticsVidhya+%28Analytics+Vidhya%29
  * The data: https://github.com/hanhanwu/Hanhan_Data_Science_Practice/blob/master/earthquake_data.csv
  * My code: https://github.com/hanhanwu/Hanhan_Data_Science_Practice/blob/master/R_animated_visualization.R
  * reference: https://www.analyticsvidhya.com/blog/2017/06/a-study-on-global-seismic-activity-between-1965-and-2016/?utm_source=feedburner&utm_medium=email&utm_campaign=Feed%3A+AnalyticsVidhya+%28Analytics+Vidhya%29
  * <b>NOTE!</b>: Before running your code, install `ImageMagick` first and then run `XQuartz`
    * To install `ImageMagick`, type `xcode-select --install`
    * Then `sudo chown -R $(whoami) /usr/local`, so that you will have the permission to do brew update
    * `brew update`
    * `sudo chown root:wheel /usr/local` change permission back
    * `brew rm imagemagick; brew install imagemagick --with-x11`
    * If all the above went well, type `xclock &` to turn on `XQuartz`. All the new Mac OS are using `Xquartz` and no longer support X11
    * <b>Run the commnds before turning on R studio</b>
    * If `XQuartz` is running, you can start to run the R code here. The visualization is very cool!
  * If after above installation, you will get many erros showing , try commands below:
    * `brew uninstall libpng`
    * `brew install libpng`
    * `brew uninstall libjpeg`
    * `brew install libjpeg`
    * `brew uninstall imagemagick`
    * `brew install imagemagick --with-x11`
    

********************************************

PYTHON PRACTICE

* Try Pandas
  * data input: https://github.com/hanhanwu/Hanhan_Data_Science_Practice/blob/master/GBM_Test.csv
  * python code: https://github.com/hanhanwu/Hanhan_Data_Science_Practice/blob/master/try_pandas.py
  * modified pyhton code: https://github.com/hanhanwu/Hanhan_Data_Science_Practice/blob/master/try_pandas.ipynb
  * I really like pandas indexing and selection methods: http://pandas.pydata.org/pandas-docs/stable/indexing.html#indexing-label
  * While implementing my modified pyhton code, I gradually felt, R is still much better than pandas in binning and ploting. R can do those data query related work such as data merging as well. But I still like Pivot Table, Multi-Indexing, Corss Tab in pandas.
  * Reference: https://www.analyticsvidhya.com/blog/2016/01/12-pandas-techniques-python-data-manipulation/?utm_content=buffer34860&utm_medium=social&utm_source=facebook.com&utm_campaign=buffer

<b>Note:</b> I'm using Spark Python Notebook, some features are unique there. Because my own machine could not install the right numpy version for pandas~


* Multi-Label Problem
  * Multi-Class vs Multi-Label
    * Multi-Class means there is only 1 column of label (dependent variable/y), and the classes of y is more than 2
    * Multi-Label means there can be multiple columns of labels, which means 1 row could have multiple labels. Also! A row could have no label at all
  * Multi-Label dataset: http://mulan.sourceforge.net/datasets.html
  * Scikit-Learn multi-label packages: http://scikit.ml/
    * To install it: `sudo easy_install scikit-multilearn` or `pip install scikit-multilearn` (Python2.*), `pip3 install scikit-multilearn` for python3.*
    * Subpackages: http://scikit.ml/api/api/skmultilearn.html
    * Scikit-Learn multi-label adapt algorithms package: http://scikit.ml/api/api/skmultilearn.adapt.html#module-skmultilearn.adapt
    * Scikit-Learn multi-label ensembling package: http://scikit.ml/api/classify.html#ensemble-approaches
  * My code: https://github.com/hanhanwu/Hanhan_Data_Science_Practice/blob/master/try_multi_label_classification.ipynb
    * If you got error "ImportError: No module named builtins", type this command in your terminal `sudo pip install future`
    * Dataset (Choose Yeast): http://mulan.sourceforge.net/datasets-mlc.html
    * Method 1 - Problem Transformation
      * Binary Relevance
      * Classifier Chains
      * Label Powerset
    * Method 2 - Adapt Algorithms
    * Method 3 - Ensembling Method in scikit-learn
      * As you can see in my code, this method had problem. I tried to install those libraries in coda emvironment and my non-virtual environment, none of them works. In order to solve that problem, you may need to install all these: https://gist.github.com/v-pravin/949fc18d58a560cf85d2
      * FInally I decided never use scikit-learn multi-label ensembling, I'd rather use normal ensembling method to predict labels using Method 1 methods. I sense, even if you installed all those, the accuracy can still be very low
   * reference: https://www.analyticsvidhya.com/blog/2017/08/introduction-to-multi-label-classification/?utm_source=feedburner&utm_medium=email&utm_campaign=Feed%3A+AnalyticsVidhya+%28Analytics+Vidhya%29


* Factorization Machines
  * Large dataset can be sparse, with Factorization, you can extract important or hidden features
  * With a lower dimension dense matrix, factorization could represent a similar relationship between the target and the predictors
  * The drawback of linear regression and logistic regression is, they only learn the effects of all features individually, instead of in combination
  * For example, you have `Fields` Color, Category, Temperature, and `Features` Pink, Ice-cream, Cold, each feature have different values
    * Linear regression: `w0 + wPink * xPink + wCold * xCold + wIce-cream * xIce-cream`
    * <b>Factorization Machines (FMs)</b>: `w0 + wPink * xPink + wCold * xCold + wIce-cream * xIce-cream + dot_product(Pink, Cold) + dot_product(Pink, Ice-cream) + dot_product(Cold, Ice-cream)`
      * dot-product: `a.b = |a|*|b|cosθ`, when θ=0, cosθ=1 and the dot product reaches to the highest value. In FMs, dor product is used to measure the similarity
      * `dot_product(Pink, Cold) = v(Pink1)*v(Cold1) + v(Pink2)*v(Cold2) + v(Pink3)*v(Cold3)`, here k=3. This formula means dot product for 2 features in size 3
    * <b>Field-aware factorization Machines (FFMs)</b>
      * Not quite sure what does "latent effects" meantioned in the tutorial so far, but FFMs has awared the fields, instead of using `dot_product(Pink, Cold) + dot_product(Pink, Ice-cream) + dot_product(Cold, Ice-cream)`, it's using Fields here, `dot_product(Color_Pink, Temperature_Cold) + dot_product(Color_Pink, Category_Ice-cream) + dot_product(Temperature_Cold, Category_Ice-cream)`, Color & Temperature, Color & category, Temperature & Category
  * `xLearn` library
    * Sample input (has to be this format, libsvm format): https://github.com/aksnzhy/xlearn/blob/master/demo/classification/criteo_ctr/small_train.txt
    * Detailed documentation about parameters, functions: http://xlearn-doc.readthedocs.io/en/latest/python_api.html
    * Personally, I think this library is a little bit funny. First of all, you have to do all the work to convert sparse data into dense format (libsvm format), then ffm will do the work, such as extract important features and do the prediction. Not only how it works is in the blackbox, but also it creates many output files during validation and testing stages. You's better run evrything through terminal, so that you can see more information during the execution. I was using IPython, totally didin't know what happened.
    * But it's fast! You can also set multi-threading in a very easy way. Check its documentation.
  * My code: https://github.com/hanhanwu/Hanhan_Data_Science_Practice/blob/master/Factorization_Machines.ipynb
    * My code is better than reference
  * Reference: https://www.analyticsvidha.com/blog/2018/01/factorization-machines/?utm_source=feedburner&utm_medium=email&utm_campaign=Feed%3A+AnalyticsVidhya+%28Analytics+Vidhya%29
  

* RGF
  * My code: https://github.com/hanhanwu/Hanhan_Data_Science_Practice/blob/master/try_RGF.ipynb
    * Looks like the evaluation result is, too bad, even with Grid Search Cross Validation
  * reference: https://www.analyticsvidhya.com/blog/2018/02/introductory-guide-regularized-greedy-forests-rgf-python/?utm_source=feedburner&utm_medium=email&utm_campaign=Feed%3A+AnalyticsVidhya+%28Analytics+Vidhya%29
    * It's missing code in the reference and it's lack of evaluation step
  * RGF vs. Gradient Boosting
    * Boosting add weights to misclassified observations for next base algorithm, in each iteration. RGF changes forest structure by one step to minimize the logloss, and also adjust the leaf weights for the entire forest to minimize the logloss, in each iteration
    * RGF searches optimum structure changes
      * The search is within the newly created k trees (default k=1), otherwise the computation can be expensive
      * Also for computational efficiency, only do 2 types of operations:
        * split an existing leaf node
        * create a new tree
      * With the weights of all leaf nodes fixed, it will try all possible structure changes and find the one with lowest logloss
    * Weights Optimization
      * After every 100 new leaf nodes added, the weights for all nodes will be adjusted. k=100 by default
      * When k is very large, it's similar to adjust weights at the end; when k is very small, it can be computational expensive since it's similar to adjust all nodes' weights after each new leaf node added
    * It doesn't need to set `tree size`, since it is determined through logloss minimizing process, automatically. What you can set is `max leaf nodes` and regularization as L1 or L2
    * RGF may has simpler model to train, compared with boosting methods, since boosting methods require small learning rate and large amount of estimators


* Regression Spline
  * Still, EXPLORE DATA first, when you want to try regression, check independent variables (features) and dependent variable (label) relationship first to see whether there is linear relationship
  * Linear Regression, a linear formula between X and y, deals with linear relationship; Polynomial Regression converts that linear formula into a polnomial one, and it can deal with non-linear relationship.
  * When we increase the power value in polynomial regression, it will be easier to become over-fitting. Also with higher degree of polynomial function, the change of one y value in the training data can affect the fit of data points far away (non-local problem). 
  * Regression Spline (non-linear method)
    * It's trying to overcome the problems in polynomial regression. When we apply a polynomial function to the whole dataset, it may impose the global data structure, so how about fit different portion of data with different functions
    * It divides the dataset into multiple bins, and fits each bin with different models
  ![regression spline](https://github.com/hanhanwu/Hanhan_Data_Science_Practice/blob/master/regression_spline_knots.png)
    * Points where the division occurs are called "Knots". The function used for each bin are known as "Piecewise function". More knots lead to the more flexible piecewise functions. When there are k knots, we will have k+1 piecewise functions.
    * Piecewise Step Functions: having a function remains constant at each bin
    * Piecewise Polynomials: each bin is using a lower degree polynomial function to fit. You can consider Piecewise Step Function as Piecewise Polynomials with degree as 0
    * A piecewise polynomial of degree m with m-1 continuous derivates is a "spline". This means:
      * Continuous plot at each knot
      * derivates at each knot are the same
      * Cubic and Natural Cubic Splines
        * You can try Cubic Spline (polinomial function has degree=3) to add these constraints so that the plot can be more smooth. Cubic Spline has k knots with k+4 degree of freedom (this means there are k+4 variables are free to change)
        * Boundrary knots can be unpredictable, to smooth them out, you can use Natural Cubic Spline
    * Choose the number and locations of knots
      * Option 1 - Place more knots in places where we feel the function might vary most rapidly, and to place fewer knots where it seems more stable
      * Option 2 - cross validation to help decide the number of knots:
        * remove a portion of data
        * fit a spline with x number of knots on the rest of the data
        * predict the removed data with the spline, choose the k with the smallest RMSE
    * Another method to produce splines is called “smoothing splines”. It works similar to Ridge/Lasso regularisation as it penalizes both loss function and a smoothing function
  * My Code [R Version]: https://github.com/hanhanwu/Hanhan_Data_Science_Practice/blob/master/learn_splines.R
  ![regression splies R](https://github.com/hanhanwu/Hanhan_Data_Science_Practice/blob/master/regression%20splines.png)
  

********************************************

DIMENSION REDUCTION

* PCA (Principle Component Analysis) - R Version
  * R code: https://github.com/hanhanwu/Hanhan_Data_Science_Practice/blob/master/PCA_practice.R
  * data set: PCA_train.csv, PCA_test.csv
  * Why using One Hot encoding to convert categorical data into numerical data and only choose the top N columns after using PCA is right
    * CrossValidated Answer: http://stats.stackexchange.com/questions/209711/why-convert-categorical-data-into-numerical-using-one-hot-encoding
    * Now I think, this question can be seperated into 2 parts. First of all, you have to convert categorical data into numerical data for PCA, with one-hot encoding, you will be able to get more info from the data. For example, one column has 100 records, among them there are 10 records as "ice-cream" while 90 records as "mochi", with this data, some algorithms will be influenced by the majority of the values ("mochi" here) and may lose accuracy. With one-hot encoding, now you generate 2 dummy columns from this 1 column, column_icecream with 1 and 0 to mark whether it's icecream or not, and column_mochi, it gives your model better chance to consider each categorical value in the original column. Secondly, after onehot encoding, you will generate many more columns. with dimensional reduction method PCA, it only selects principle components (the drawback is, it changed your data and made it non-interpretable, may loose some info) which will represent the major info from the original data, could make your model faster and even getting higher accuracy

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
  * <b>When using PCA, it is better to scale data in the same unit</b>


* Easy simple way to do feature selection with Boruta (so far it's the most convenient feature selection method I have tried):
  * R code: https://github.com/hanhanwu/Hanhan_Data_Science_Practice/blob/master/feature_selection_Boruta.R
  * Training data: https://github.com/hanhanwu/Hanhan_Data_Science_Practice/blob/master/loan_prediction_train.csv
  * Testing data: https://github.com/hanhanwu/Hanhan_Data_Science_Practice/blob/master/loan_prediction_test.csv
  * Tutorial: http://www.analyticsvidhya.com/blog/2016/03/select-important-variables-boruta-package/?utm_content=bufferec6a6&utm_medium=social&utm_source=facebook.com&utm_campaign=buffer


* t-SNE, dimension reduction for non-linear relationship
  * reference: https://www.analyticsvidhya.com/blog/2017/01/t-sne-implementation-r-python/?utm_source=feedburner&utm_medium=email&utm_campaign=Feed%3A+AnalyticsVidhya+%28Analytics+Vidhya%29
  * practice dataset for R code: https://github.com/hanhanwu/Hanhan_Data_Science_Practice/blob/master/tsne_csv.zip
  * R code: https://github.com/hanhanwu/Hanhan_Data_Science_Practice/blob/master/t-SNE_practice.R
  * NOTE: my plot results are different from those perfect clusters shown in the tutorial, I guess the author was using another dataset for visualization, but want to show how to use t-SNE... 
  * Python code (t-SNE works pretty well on the dataset used here): https://github.com/hanhanwu/Hanhan_Data_Science_Practice/blob/master/t-SNE_Practice.ipynb
  
  
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
  
* Deal with Data Shifting
  * My Code: https://github.com/hanhanwu/Hanhan_Data_Science_Practice/blob/master/deal_with_data_shifting.ipynb
  * This can be helpful to check `training vs testing` data or `old vs new` data or `time period x vs time period y` data
  * Check and deal with it before working on your final model prediction
  * reference: https://www.analyticsvidhya.com/blog/2017/07/covariate-shift-the-hidden-problem-of-real-world-data-science/?utm_source=feedburner&utm_medium=email&utm_campaign=Feed%3A+AnalyticsVidhya+%28Analytics+Vidhya%29
  * In method 2, I used MLBox
    * https://github.com/AxeldeRomblay/MLBox
    * This tool does data cleaning, removing shifting data for you, after doing each preprocessing, it gives you statistics for the data
  * Method 1 used here has more flexible and maybe more accurate, since you don't know what kind of data cleaning work that MLBox did for you
 
* Deal with Imbalanced Dataset
  * practice 1: https://github.com/hanhanwu/Hanhan_Data_Science_Practice/blob/master/deal_with_imbalanced_dataset.R
  * practice 2: https://github.com/hanhanwu/Hanhan_Data_Science_Practice/blob/master/deal_with_imbalanced_data_2.R
  * training data for practice 2: https://www.analyticsvidhya.com/wp-content/uploads/2016/09/train.zip
  * testing data for practice 2: https://www.analyticsvidhya.com/wp-content/uploads/2016/09/test.zip
  * data description for practice 2: http://archive.ics.uci.edu/ml/machine-learning-databases/census-income-mld/census-income.names
  * Tutorial 1: https://www.analyticsvidhya.com/blog/2016/09/this-machine-learning-project-on-imbalanced-data-can-add-value-to-your-resume/?utm_source=feedburner&utm_medium=email&utm_campaign=Feed%3A+AnalyticsVidhya+%28Analytics+Vidhya%29
  * Tutorial 2: https://www.analyticsvidhya.com/blog/2017/03/imbalanced-classification-problem/?utm_source=feedburner&utm_medium=email&utm_campaign=Feed%3A+AnalyticsVidhya+%28Analytics+Vidhya%29
  * Summarize Methods in dealing with Data Imbalance
    * Random Undersampling - may lose important data, but faster
    * Random Oversampling - May have overfitting
    * Cluster based Oversampling - Cluster each class first. Then for each class, all the clusters oversampled to have same number of data points. This method overcomes the data imbalance between and within classes. However, overfitting can still happen.
    * SMOTE (Synthetic Minority Over-sampling Technique) - it generates similar data based on the chosen subset of minority class data. Therefore it could somewhat overcome the overfitting caused by duplication in the above oversampling methods. But, it may create overlapped data in both minority and majority classes, and therefore creates noise. It does not consider the underlying distribution of the minority class
    * MSMOTE (Modified synthetic minority oversampling technique) - Modified based on SMOTE, it classifies the samples of minority classes into 3 distinct groups – <b>Security/Safe samples, Border samples, and latent noise samples</b>, by calculating the distances among samples of the minority class and samples of the training data. Security samples are those data points which can improve the performance of a classifier. Noise are the data points which can reduce the performance of the classifier.  The ones which are difficult to categorize into any of the two are classified as border samples. SMOTE randomly selects a data point from the k nearest neighbors for the security sample, selects the nearest neighbor from the border samples and does nothing for latent noise.
    * <b>Use SMOTE/MSMOTE with GBM, XGBoost, Random Forest may be better</b>
  * Emsembling
    * Bagging - allows sampling with repalcement (bootstrap). Random Forest is a type of Bagging. Bagging improves stability & accuracy of machine learning algorithms; Reduces variance； Overcomes overfitting； Improved misclassification rate of the bagged classifier； In noisy data environments bagging outperforms boosting； Bagging works only if the base classifiers are not bad to begin with. Bagging bad classifiers can further degrade performance
    * <b>Weak Learner</b> - indicates the classifier works slightly better than the average
    * Boosting, Ada Boost - Suits for any kind of classification problem; Not prone to overfitting; Sensitive to noisy data and outliers
    * Boosting, Gradient Tree Boosting (GBM) - uses Decision Tree as weak learner; each model minilize loss function using Gradient Descent; harder to fit than random forest; 
    * Boosting, XGBoost


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
* [Python] LightGBM Practice
  * For more resources, check LightGBM here: https://github.com/hanhanwu/Hanhan_Data_Science_Resources2/blob/master/README.md
  * data: http://archive.ics.uci.edu/ml/datasets/Adult
  * Install python xgboost to compare with LightGBM: http://xgboost.readthedocs.io/en/latest/build.html
    * Default is without multi-threading
    * It also mentions how to install xgboost with multi-threading
  * PRACTICE 1 - Single Thread LightGBM vs Single Thread XGBoost - no cross validation: https://github.com/hanhanwu/Hanhan_Data_Science_Practice/blob/master/try_lightGBM_single_thread_basics.ipynb
    * This one is a basic one, only use single thread, even without cross validation.
    * When you are using `train()` method, it does not allow you to set seed, and each time you may get different results.
    * reference: https://www.analyticsvidhya.com/blog/2017/06/which-algorithm-takes-the-crown-light-gbm-vs-xgboost/?utm_source=feedburner&utm_medium=email&utm_campaign=Feed%3A+AnalyticsVidhya+%28Analytics+Vidhya%29
  * XGBoost paper: https://github.com/hanhanwu/Hanhan_Data_Science_Practice/blob/master/xgboost_paper.pdf
    * The reason I'm checking this paper is because XGBoost has its regularizer, meanwhile I was thinking how to use the `cv()` function and whether `cv()` is needed if XGBoost has its built-in regularizer. My answer is, yes, cross validation is still necessary. regularizer is used for reducing overfitting.
    * According to the paper, if you set regularizer params all as 0, XGBoost works the same as GBM
    * Boosting methods such as GBM tend to overfitting if you don't use cross validation or regularization. However, according to my notes, XGBoost has regularizer to reduce overfitting
    * my notes: https://github.com/hanhanwu/Hanhan_Data_Science_Resources/blob/master/Experiences.md
    * In python xgboost API, you will find both classifier and regressor have 2 regularizer params, `reg_alpha` and `reg_lambda`. http://xgboost.readthedocs.io/en/latest/python/python_api.html#module-xgboost.training
  * PRACTICE 2 - Add cross validation
    * Both LightGBM and XGBoost training API has `cv()` method to do cross validation, however, it only show you the evaluation results for each cv round, wihtout giving you the best set of parameters. In fact, it only uses the set of params you defined before `cv()`.
    * So, I started to try their scikit-learn wrapper, which has cross validation to help choosing the best set of params
    * <b>reference - tuning python xgboost [Python]</b> (feature importance plot part not work for me, better check my code below): https://www.analyticsvidhya.com/blog/2016/03/complete-guide-parameter-tuning-xgboost-with-codes-python/
    * <b>NOTE:</b> Whwn you try to tune the params with `GridSearchCV`, it's better to seperate params into a few sets, and tune each set to ginf optimum valus one by one. Otherwise, it can be running forever.
    * my code: https://github.com/hanhanwu/Hanhan_Data_Science_Practice/blob/master/try_lightfGBM_cv.ipynb
    * <b>NOTE:</b> When you are using LightGBM with GridSearchCV, `n_jobs` has to be set as 1, otherwise, the code will run forever
    * As you can see in the code, both XGBoost and LightGBM have 2 cross validation methods. 1) If you use their built-in `cv()` method, the nuber of thread depends on how did you install XGBoost and how did you make LightGBM with `make -j`. Meanwhile, this type of cross validation could only allow you to find see optimum `n_estimators`, and it's not convenient. 2) If you use `GridSearchCV()`, you can define multi-threading in the code, meanwhile it finally will return optimum param set, if you turn params subgroup by subgroup
    * During param tuning with GridSearchCV, LightGBM didn't appear to be faster, maybe because `n_jobs` has to be 1 for while in XGBoost, I used 7. But later in model training, it's faster than XGBoost
    * In the experiments, you will see, even with cross validation, regularization and params that help avoid overfitting, overfitting could still happen
  * Summary for Using <b>CPU</b> with XGBoost and LightGBM
    * When you are installing XGBoost or `make` LightGBM, you can set whether it's multi-threading or single thread. In this step, LightGBM is more convenient, since you just type `make -j [num]`, change num to decide how many thread do you want, while XGBoost requires to reinstall
    * When you are using `train()` or `cv()` provided by both algorithms, you cannot get a set of optium parameters
    * If you want to tune a set of parameters, use scikit-learn `GridSearchCV()` for the tunning. For XGBoost, it allows you to set number of threads for XGBoost and number of jobs runninng in parallel for grid search; However, for LightGBM, you can set multiple threads for LighGBM but the `n_jobs` has to be 1.
  * Other Resources
    * XGBoost Params (including R package): http://xgboost.readthedocs.io/en/latest/parameter.html#general-parameters
    * Complete guide in tunning GBM: https://www.analyticsvidhya.com/blog/2016/02/complete-guide-parameter-tuning-gradient-boosting-gbm-python/
    * [R] - LightGBM R package: https://github.com/Microsoft/LightGBM/tree/master/R-package
    * LightGBM params: http://lightgbm.readthedocs.io/en/latest/Parameters-tuning.html
    * scikit-learn GridSearchCV: http://scikit-learn.org/stable/modules/generated/sklearn.model_selection.GridSearchCV.html
  * <b>NOTE: Check param documents and 2 complete param tuning guide of XGBoost and LightGBM above, start params with typical value range for tuning</b>


********************************************

ADVANCED TOOLS

* TPOT
  * https://rhiever.github.io/tpot/
  * It does feature selection, model selection and param optimization automatically. It uses genetic algorithm to optimize the parameters
  * My code: https://github.com/hanhanwu/Hanhan_Data_Science_Practice/blob/master/try_genetic_alg_through_TPOT.ipynb
    * scoring method: By default, accuracy is used for classification and mean squared error (MSE) is used for regression. https://rhiever.github.io/tpot/using/#scoring-functions
    * At the end of its pipelien output, you will see selected model with optimized params
  * Reference: https://www.analyticsvidhya.com/blog/2017/07/introduction-to-genetic-algorithm/?utm_source=feedburner&utm_medium=email&utm_campaign=Feed%3A+AnalyticsVidhya+%28Analytics+Vidhya%29


* MLBox
  * http://mlbox.readthedocs.io/en/latest/introduction.html
  * It does some data cleaning, removing shifting features, model selection and param optimization automatically, and each step, it output useful information or output the final predicted results, as well as CPU time
  * For model selection and param optimization, it functions similar to TPOT
  * My code: https://github.com/hanhanwu/Hanhan_Data_Science_Practice/blob/master/try_mlbox.ipynb
  * Reference: https://www.analyticsvidhya.com/blog/2017/07/mlbox-library-automated-machine-learning/?utm_source=feedburner&utm_medium=email&utm_campaign=Feed%3A+AnalyticsVidhya+%28Analytics+Vidhya%29
  

* CatBoost
  * reference: https://www.analyticsvidhya.com/blog/2017/08/catboost-automated-categorical-data/?utm_source=feedburner&utm_medium=email&utm_campaign=Feed%3A+AnalyticsVidhya+%28Analytics+Vidhya%29
    * I changed a lot of its code
  * My basic code: https://github.com/hanhanwu/Hanhan_Data_Science_Practice/blob/master/try_CatBoost_basics.ipynb
    * If you set output dir, you should get these: https://tech.yandex.com/catboost/doc/dg/concepts/output-data-docpage/#output-data
    * The way CatBoost define "feature importance" looks special. I prefer the feature importance generated by Random Forest, XGBoost or LightGBM. https://tech.yandex.com/catboost/doc/dg/concepts/fstr-docpage/#fstr
    * You can save the trained model, and later reload this pre-trained model
    * As you can see, the data and the preprocessing step is quite similar to the code in TPOT: https://github.com/hanhanwu/Hanhan_Data_Science_Practice/blob/master/try_genetic_alg_through_TPOT.ipynb
    * But with CatBoost, it does categorical features' label encoding in statistical methods. For me, in real world practice, I will still try other categorical to numerical methods, such as one-hot encoding.


*********************************************

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

