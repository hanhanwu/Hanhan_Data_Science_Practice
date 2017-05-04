
Recently, I did many experiments on classification with imbalanced small dataset. Although I still have many experiments in plan before Dec. 31 2016, it's time to record my summary here.


***********************************************************************

<b>Overview</b>

* The dataset I used here is very small, only around 6000+ records and it has severe data imbalance problem. Because of privacy issue, I cannot share anything detail about the data. But I need to write donw the summary to help my future data science work, since by doing these experiments, I am learning a lot.
* In my current experiments, I have tried different data preprocessing methods/libraries, along with my favorite classifcation algorithms. The results were quite different from what I have learned from online tutorials but it is very interesting.
* Here I am storing my sample code for data explore process, data preprocessing, feature selection wih different libraries and model training & evaluaion.


***********************************************************************

<b>Very First Things I did</b>

* After complex data collection and data integration, I saved the dataset in the database. Because the data collection work in this real world project was terribly time consuming.... And also, for SQL Server, R database connector could only connect to one Database per handler, (if it's Oracle, one handler is able to connect to all the Databases in a Server). I had to use SQL Server.
* After reading the data through R `fread`, the data became a `data.table`. R data.table is great in multiple operation and it's much faster for data loading when the dataset is very large.
* Then, I removed all the ID columns.
* One thing need to note is, R data table is a reference, when you copy a data table, both the original and the copy point to the same location, which means when you are chaning the copy, the original will be changed too. This is opposite to R data frame. So, if you really want to create a copy of a data.table, copy its data.frame.  
* My code: https://github.com/hanhanwu/Hanhan_Data_Science_Practice/blob/master/classification_for_imbalanced_data/first_of_all.R


***********************************************************************

<b>Data Explore</b>

* There are 200+ features, but I did data exploration by checking feature by feature, at the same time, I was recording potential features that need to deal with outliers, or should using binning to reduce levels, or should using one-hot encoding, or the feature is almost 0 variance and can be removed.
* About data exploration, I really like this post: https://www.analyticsvidhya.com/blog/2016/01/guide-data-exploration/
* However, when working on real world data, even if we do all the steps mentioned in the above post, we may not get the optimal results, and sometimes, at least in my experiments, some methods brounght much worse results.
* I did univariate and bivariate for each feature, this is necessary, not only to know the data better, but also to deal with <b>data format transforamtion, data skewness, missing data and 0 variance data</b> along the way; and record features that need further data preprocessing
* When checking data with missing data, the distribution could help you understand whether the data is random missing. If it's not random missing, I either keep the feature for later imputing or binning the data so that I won't miss any possible important data. If it's random missing, I keep features that have missing percentage within a range, and drop those features have too many randomly missing values.
* To deal with data skewness, we can use `log` or `sqrt` depends on the data central tendency, the result should be closer to normal distribution.
* Sometimes, distribution plot may not be enough, with `quantile` and boxplot, it is better to check data central tendency and outliers.
* In my case, the target is categorical, so for numerical, instead of calculating their z-score or do anova test, I just plot the density for each target value and put the plots together, this is easier to understand.
* <b> NOTE: </b> Later in model training, algorithms like Random Forests and xgboost all need numerical features, so even if from business point of view, the data belongs to categorical data, but after data loading it has been defined as numerical data, there is no need to convert the data to categorical if you won't do any other process, otherwise later you still have to convert it to numerical and the values will be changed, and the final prediction results won't be improved.
* My code: https://github.com/hanhanwu/Hanhan_Data_Science_Practice/blob/master/classification_for_imbalanced_data/data_explore.R


***********************************************************************

<b>Data Preprocessing</b>

* Here, I am listing the general methods for data preprocessing can be used all the time before training the model.
* When dealing with missing data, I used 2 methods here. One is to use median/mode based on central tendency, the other is to use KNN to predict the missing values, meanwhile Caret KNN will help you normalize the numerical data at the same time, but if there are categorical data, it ignores them. Based my several rounds of experiments, for the dataset I am using, KNN + data normalization always gave me better results.
* One thing need to note about using KNN data imputing method, it only deals with numerical data and the 0 variance numerical data should be removed before using KNN.
* About missing data imputing, I also tried to replace NA with "MISSING" just in case missing data could in fact help the prediction, this method worked well in some of my other projects.
* Then it's helpful to remove those 0 variance data and highly correlated features. Because 0 variance data cannot contribute anything, highly correlated features will increate the data variance.
* Dealing with outliers, I wrote 2 methods, one is to use median/mode based on the central tendency, the other is to binning the data because sometimes, we don't want to lose any information expecialy when the data is small. However, for the dataset I am using here, it turned out that dealing with outliers gave me a lower balanced accuracy.
* Besides 0 variance features, some features can be almost constant, therefore, in my code, I wrote a fnction to find these almost constant features. I removed those with no more then 4 distinct values, however, for the dataset I am using here, the balanced accuracy dropped a lot. Therefore, I think for small dataset with severly imbalance problems, removing almost constant features may not be a good idea.
* My code: https://github.com/hanhanwu/Hanhan_Data_Science_Practice/blob/master/classification_for_imbalanced_data/general_data_preprocessing.R


***********************************************************************

<b>Feature Importance - Feature Selection</b>

* 3 major feature selection methods: https://www.analyticsvidhya.com/blog/2016/12/introduction-to-feature-selection-methods-with-an-example-or-how-to-select-the-right-variables/?utm_source=feedburner&utm_medium=email&utm_campaign=Feed%3A+AnalyticsVidhya+%28Analytics+Vidhya%29
* In my code, I have used filter methods and wrapper methods.
* For filter methods, I tried gain.ratio, information.gain (1-entropy) and anova.test, however, after I took selected features in to models, all got very low balanced accuracy, especially for the small class. This is because filter methods are trying to calculate the correlation between features and the target, I guess when the target is serverly imbalanced, the secected features may all have bias toward the large class.
* For wrapper methods, I tried Caret package and Boruta, both of them use Random Forests as default, however the final balanced accuracy had significant difference. Caret feature selection is recursive methods, Boruta feature selection is all-relevant selection. Boruta gave me much higher balanced accuracy for this small and imbalanced accuracy. However, one thing I like Caret feature importance is, after model training, not only random forests, algorithms such as GBM can also plot the final feature importance.
* For more about Caret and Boruta, you can find resources here: https://github.com/hanhanwu/Hanhan_Data_Science_Resources2
* For embedded methods, I think ensembling methods like xgboost is doing that for you.
* In my experiments, I have even tried to use regression for feature selection, selecting those with higher coefficient. Did not work well for this dataset
* My code: https://github.com/hanhanwu/Hanhan_Data_Science_Practice/blob/master/classification_for_imbalanced_data/feature_selection.R
* The paper about Boruta and all-relevant feature selection definition: https://github.com/hanhanwu/Hanhan_Data_Science_Practice/blob/master/classification_for_imbalanced_data/boruta_all_relevant_feature_selection.pdf


***********************************************************************

<b>Model Training and Evaluation</b>

* In my experiments, I have found that bagging method Random Forests, Boosting methods GBM, XGBOOST and C50 could always return higher balanced accuracy. I used ROSE to deal with data imbalance problem and overcome the shortage of overfitting, underfitting. However it dind't work well most of the time... Random Forests is pretty great. In fact, it can handle missing data, outliers, data imbalance itself well.
* Later I found that SVM is good for small dataset, especially when the number of featues are large, even larger than the number of data records. in somd e of my experiments in small dataset, SVM even performs better than Random Forests and other algorithms. For SVM, need to turn 3 major parameters, `C, gamma and kernel`. For details, please check my notes here: https://github.com/hanhanwu/Hanhan_Data_Science_Resources/blob/master/Experiences.md
* Since the data is imbalnced, I am using Balanced Accuracy, Sensitivity and Specifity as the measure
* In the code, I have also tried to tune the Threshold and see whether the balanced accuracy could be improved.
* My code: https://github.com/hanhanwu/Hanhan_Data_Science_Practice/blob/master/classification_for_imbalanced_data/model_training.R
* About model evaluation, I tried Balanced Accuracy, AUC and Reliability Plot, among them, Reliability Plot is the visualized calibration, it is supposed to capture those cannot be captured by AUC and Accuracy, but my code here may not be right, the visualization looks weird.
* My code - 3 Classification Model Evaluation Methods: https://github.com/hanhanwu/Hanhan_Data_Science_Practice/blob/master/classification_for_imbalanced_data/3_classification_evaluation_methods.R


***********************************************************************

<b>Feature Engineering - One Hot Encoding</b>

* I was about to give up using one-hot encoding, because there are too many categorical data and many has higher levels, checking each of them with the target in order to know how to reduce the levels can be time consuming. But, with all my patience, I still did it. It's worthy. The Balanced Accuracy has increased from 0.94469 to 0.94793, especailly the accuracy for small class has reached to be higher than 90%, and the final feature importance makes more sense.
* In order to do feature engineering, I need to check each feature and change data format, then reduce the levels of high levels categorical data, and use one-hot on categorical data.
* Impute all the missing data and use Boruta, since Boruta requires no NA in the data.
* My code (in my code you will find all the examples from one-hot to model training & evaluation): https://github.com/hanhanwu/Hanhan_Data_Science_Practice/blob/master/classification_for_imbalanced_data/one_hot_encoding.R


***********************************************************************

<b>Feature Engineering - Clustering for Classification</b>

* It is amazing to try clustering in a classification problem. And I'm just using Kmeans! The balanced accuracy improved from 0.94793 to 0.95067.
* I did all the data preprocessing, then used One-Hot Encoding and Boruta Feature Selection. With the rest features, I used kmeans to generate cluster. The generated cluster ids formed a new feature. With this added new feature among all the selected features, I trained the model with Random Forests.
* If you want to get the highest balanced accuracy, may need to write an iteration by changing k of kmeans, but here Ranfom Forests ran very slow because I was doing param tuning with cross validation. So, I dind't do that here.
* My code: https://github.com/hanhanwu/Hanhan_Data_Science_Practice/blob/master/classification_for_imbalanced_data/clustering_in_classification.R
* When doing clustering, with grounf truth, we call this as Intrinsic Clustering. With the ground truth, it can be easier to find the optimal number of clusters. 
* Reference (I think that person who wrote 8 methods, is crazy): http://stackoverflow.com/questions/15376075/cluster-analysis-in-r-determine-the-optimal-number-of-clusters
* code for finding optimal number of clusters: https://github.com/hanhanwu/Hanhan_Data_Science_Practice/blob/master/classification_for_imbalanced_data/find_optimal_clusters.R

* More about clustering
  * My code: https://github.com/hanhanwu/Hanhan_Data_Science_Practice/blob/master/classification_for_imbalanced_data/clustering2.R
  * In my code, you will find:
    * elbow method for k-means, in order to find optimal cluster number
    * silhouette coefficient, to check cluster similarity, higher the better
    * hierarchical clustering
    * Clusering Emsembling with hierarchical clusters


***********************************************************************

<b>Notes</b>

* Methods like Boruta feature sdelection and clustering used here do not guarantee to improve balanced accuracy in all the time, sometimes only using Ranfom Forests for small imbalanced dataset can achiveve highest balanced accuracy, and Random Forests can deal with missing data, outliers and data imbalance itself.
* It can be helpful add much more features. I added 100+ new features into the dataset, with data preprocessing, Boruta features selection and methods such as cross validation to avoid overfitting, my balanced accuracy improved from 0.95067 to 0.96573, especially for Sensitivity, which jumped from 0.90698 to 0.93478. But using one-hot encoding does not always guarantee the improvement.
* <b>Another whole code for imbalance data can be found here</b>: https://github.com/hanhanwu/Hanhan_Play_With_Social_Media/blob/master/Predict_StackOverflow_Underrated_Answers/data_analysis_all_code.R
  * The data here is not only imbalanced, but also very small
  * I put the new code parts here in the above sections too
  * I have learned:
    * GBM may work better with SMOTE than XGBoost
    * SMOTE generate similar data, so I'd better not use it on testing data
    * ROSE copies data, so I could use it on both training and testing data
    * When using k-means clustering, it's better to change seed multiple times in order to get more optimal results
    * Sometimes using hierarchical clustering is better than k-means, since it doesn not require you to set k
    * When traditional clustering does not work well, the data maybe non-convex, try <b>spectral clustering</b>.
