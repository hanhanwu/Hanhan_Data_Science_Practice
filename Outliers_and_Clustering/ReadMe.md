Outliers Detection and Clustering are related to each other, and in a world without groud truth data, it requires more efforts to find effective and efficient methods.


*********************************************************************************

LEARNING NOTES

* As [my data mining bible reading notes][1] recorded (Chapter 11, high demensional clustering), even it's biclustering (such as <b>MaPle</b>), which searches subsapace in both objects and features, can be time consuming because it enumerates all the subspaces. Here is another implemented example of [HiCS, LOF with description][2], [HiCS code only][3], according to the author, HiCS can solve the subspaces search in an more effective way. I think so, since MaPle is publised in 2008, HiCS is published in 2012. So deserve to try
* People have also implemneted another method to detect outliers, [LOF][4], it is densitiy based, and calculates nearest neighbours. Note that, when there is high dimensional features, PCA (linear regression model), LOF (proximity-based) can be less effective. This is the so-called The Curse of Dimensionality, when there are more dimensions, these methods can be time consuming, and outliers, random noise could make the calculation results lose meaning


*********************************************************************************

PRACTICE CODE

* Select Optimal Number of Clusters
  * My code: https://github.com/hanhanwu/Hanhan_Data_Science_Practice/blob/master/Outliers_and_Clustering/clustering_evaluation.ipynb
  * Elbow Method
    * Useful when the data can be well clustered
    * Choose the point where the inside angle is the smallest
    * reference: https://bl.ocks.org/rpgove/0060ff3b656618e9136b
  * Silhouette Score
    * Higher the score, the data is better clustered. So when choosing the optimal k, choose the one give the highest silhouette score
    * But similar to k-means (can only find convex clusters), silhouette score is higher for convex clusters than other types of clusters (such as density based clusters which obtained from DBSCAN)
* Clustering Performance Measurement
  * The performance measurements are directly linked to finding the optimal number of clusters
  * Beside the method above, I found sklearn summary about its available measurements is pretty good: http://scikit-learn.org/stable/modules/clustering.html#clustering-performance-evaluation
    * Not only the code, but also advantage and disadvantages

* BiClustering - HiCS vs LOF
  * Reference Tutorial (code is not formated): http://shahramabyari.com/2016/01/19/detecting-outliers-in-high-dimensional-data-sets/
  * Reference Code: https://github.com/shahramabyari/HiCS/blob/master/hcis.py
  * My data: https://github.com/hanhanwu/Hanhan_Data_Science_Practice/blob/master/PCA_test.csv
  * My code: https://github.com/hanhanwu/Hanhan_Data_Science_Practice/blob/master/HiCS_biclustering.ipynb
  * Challenge: My godness, it's so troublesome to use these scholar implemented code.... They want to publish the paper, and code is designed just for that type of data. When I am using a randaom set of other data, took me so much time to modify the code and still, has warnings, although I got the results...


* Pseudo Labeling - A type of semi-clustering
  * How pseudo labeling work
    1. train your training dataset d1
    2. use the trained model to predict a new dataset d2, the predicted label is the pseudo label
    3. combine both d1 (with its label) and d2 (with pseudo label) as training data, train the model again
    * According to theory, adding more data with pseudo labels will improve accuracy
  * My code: https://github.com/hanhanwu/Hanhan_Data_Science_Practice/blob/master/Outliers_and_Clustering/pseudo-labeling.ipynb
    * In this code, I used TPOT to find best model and optimize the model automatically. Because we don't have the ground truth for testing data, we could check MSE during the validation period. As you can see, by using pseudo labeling, MSE reduced 10 times
    * NOTE: TPOT is built on scikit-learn. This is a regression problem, validation is using k-fold as default just like scikit-learn; if it's a classification problem, it will be stratified k-fold in order to distribution classes to each sample in almost the same percent
  * reference: https://www.analyticsvidhya.com/blog/2017/09/pseudo-labelling-semi-supervised-learning-technique/?utm_source=feedburner&utm_medium=email&utm_campaign=Feed%3A+AnalyticsVidhya+%28Analytics+Vidhya%29
    * The code in this article is too complex, I'm not referencing it at all
    
* Subspace Clustering
  * It focuses on localized clustering, used to help remove items that do not belong to the cluster
  * This time, I tried R Notebook: https://github.com/hanhanwu/Hanhan_Data_Science_Practice/blob/master/Outliers_and_Clustering/try_subsapce_clustering.Rmd
  * It will generate a html file: https://github.com/hanhanwu/Hanhan_Data_Science_Practice/blob/master/Outliers_and_Clustering/try_subsapce_clustering.nb.html
    * It seems that GitHub cannot show the output directly. Meanwhile, the plot in this code is dynamic, it can be seen and interact with in R studio, but won't be shown in the R notebook

*********************************************************************************

IDEAS SPARK

* About data labeling
  * When there are many data records, and you don't have the label, meanwhile you are not human expert to label all the data right
    * Semi Clustering, then check each cluster, labeled and unlabeled data
    * 2 people to label the data, but need some overlapped data from each person, calculate Cohen's kappa coefficient, higher the coefficient, better
    * 2+ people to label the data, but need some overlapped data from each person, calculate Fleiss's kappa coefficient, higher the coefficient, better
    * Crowd Sourcing labeling, but either you calculate Fleiss's kappa, or take the risk of accuracy. This will be my last choice in industry


[1]:https://github.com/hanhanwu/readings/blob/master/ReadingNotes_DMBible.md
[2]:http://shahramabyari.com/2016/01/19/detecting-outliers-in-high-dimensional-data-sets/
[3]:https://github.com/shahramabyari/HiCS/blob/master/hcis.py
[4]:http://shahramabyari.com/2015/12/30/my-first-attempt-with-local-outlier-factorlof-identifying-density-based-local-outliers/
