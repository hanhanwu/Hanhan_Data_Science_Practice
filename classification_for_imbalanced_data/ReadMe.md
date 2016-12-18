
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
* In my case, the target is categorical, so for numerical, instead of calculating their z-score or do anova test, I just plot the density for each target value and put the plots together, this is easier to understand
* My code: https://github.com/hanhanwu/Hanhan_Data_Science_Practice/blob/master/classification_for_imbalanced_data/data_explore.R
