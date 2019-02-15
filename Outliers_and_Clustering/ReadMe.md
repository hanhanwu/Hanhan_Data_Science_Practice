Outliers Detection and Clustering are related to each other, and in a world without groud truth data, it requires more efforts to find effective and efficient methods.


## LEARNING NOTES

* As [my data mining bible reading notes][1] recorded (Chapter 11, high demensional clustering), even it's biclustering (such as <b>MaPle</b>), which searches subsapace in both objects and features, can be time consuming because it enumerates all the subspaces. Here is another implemented example of [HiCS, LOF with description][2], [HiCS code only][3], according to the author, HiCS can solve the subspaces search in an more effective way. I think so, since MaPle is publised in 2008, HiCS is published in 2012. So deserve to try
* People have also implemneted another method to detect outliers, [LOF][4], it is densitiy based, and calculates nearest neighbours. Note that, when there is high dimensional features, PCA (linear regression model), LOF (proximity-based) can be less effective. This is the so-called The Curse of Dimensionality, when there are more dimensions, these methods can be time consuming, and outliers, random noise could make the calculation results lose meaning

### Clustering Evaluation Methods
* NMI (Normalized Mutual Information)
  ![NMI](https://github.com/hanhanwu/Hanhan_Data_Science_Practice/blob/master/Outliers_and_Clustering/NMI.png)
    * It measures the purity of Ci by calculating the largest number of common objects that cluster Ci has with all the other mutual clusters Mi. Higher NMI, the higher purity of Ci is.
* We have NMI to measure between cluster similarity, we should also measure the within ckuster purity to guarantee the ckustering quality
  
### PyOD - An Outlier Detection Python Library
* All the algorithms it supports: https://pyod.readthedocs.io/en/latest/
* IPython Notebook - compare all algorithms: https://github.com/yzhao062/pyod#api-cheatsheet--reference
  

## PRACTICAL CODE

* Data Exploration - Visualized Projected Clusters
  * My code : https://github.com/hanhanwu/Hanhan_Data_Science_Practice/blob/master/Outliers_and_Clustering/dimensional_reduction_visualization.ipynb
    * It uses T-SNE to show 2D, 3D reduced dimensional projections
    * T-SNE is famous for 2D visualization, but better for lower dimensional data
  * The methods used here are not clustering, they are dimensional reduction in sklearn `manifold`. http://scikit-learn.org/stable/modules/manifold.html
    * All the methods in this library are based on nearest neighbour search, therefore scalling all the features are necessary.
    * The basic idea behind these dimensional reduction is majorly about project the original data to a lower dimension. Therefore, the data I ploted in the code is no longer the original data. They are projected data
  * The reason I put this method here while the methods are not clustering, is to record a way to show the audiance that what does projected clusters can look like. Although it's not real clustering that could produce you the results, it can give you some insights. Working in the industry, facing those audiance who knows nothing about data science, you really need some easy to understand visualization

* Select Optimal Number of Clusters
  * My code [Python]: https://github.com/hanhanwu/Hanhan_Data_Science_Practice/blob/master/Outliers_and_Clustering/clustering_evaluation.ipynb
  * Direct Methods vs Testing Methods
    * Direct methods consists of optimizing a criterion, such as the within cluster sums of squares or the average silhouette. Such as elbow method and silhouette score
    * Testing Methods consists of comparing evidence against null hypothesi. Such as Gap Statistics
  * Elbow Method
    * The idea behind to to choose the k that can minimize the total within-cluster sum of square, or you can consider it's trying to minimize the dispersion within clusters and maximize the dispersion between clusters - compatness of the clusters
    * Useful when the data can be well clustered
    * Choose the point where the inside angle is the smallest
    * reference: https://bl.ocks.org/rpgove/0060ff3b656618e9136b
  * Silhouette Score
    * It measures the quality of clusters by determining how well each object lies within its cluster.
    * Higher the score, the data is better clustered. So when choosing the optimal k, choose the one give the highest silhouette score
    * But similar to k-means (can only find convex clusters), silhouette score is higher for convex clusters than other types of clusters (such as density based clusters which obtained from DBSCAN)
  * Gap Statistics
    * Similar to Elbow Method, it uses total within-cluster sum of square (total intracluster variation), the difference is it has reference dataset generated using Monte Carlo simulations. That is, for each variable (xi) in the dataset it computes its range [min(xi),max(xj)] and generate values for the n points uniformly from the interval min to max.
    * With both reference dataset and observed dataset, Gap Statistics calculates the gap between reference data total intracluster variation and observed data total intracluster variation. The larger the gap is, the better clustering is.
    * Optimal k has the largest gap score.
    * I think it also only work for convex clusters
    * <b>I don't recommend to use current python gap statistics for now</b>
      * The implemetation is here: https://anaconda.org/milesgranger/gap-statistic/notebook
      * The optimal k tend to be too large.
      * That open source code also has other problem, such as didn't set seed. Plus, the package has to use python3 to install
  * <b>Strongly recommend to use R to find optimal number of k for clustering!</b>
    * My code [R]: https://github.com/hanhanwu/Hanhan_Data_Science_Practice/blob/master/Outliers_and_Clustering/finding_optimal_k.R
      * First of all, with package `NbClust`, you can use 30 algorithms to calculate optimal k, and choose the k with highest votes
      * Package `factoextra` also allows you to use elbow, silhouette with 1 line of code, gap statistics  needs 2 lines of code. I still don't like gap statistics, since different seed could totally change the optimal k. The visualization is pretty good, there is a line to mark the best k, I think this is especially useful when it's difficult to choose k from elbow method visualization
      * In the code, it tried k-means, PAM (k-modoids) and hierarchical clustering for the same dataset, with different methods.
      * I think using 30 algorithms and choose highest voted k is the most reliable way. If we have to use 1 method, I prefer Silhouette Score, since in the visualization is easier to see, you just choose the k with highest score. Elbow method maybe difficult to see when the data is not well clustered. Gap statistics always seems unstable to me...
    * Refernece: http://www.sthda.com/english/wiki/print.php?id=239
* sklearn Clustering Performance Measurement
  * The performance measurements are directly linked to finding the optimal number of clusters
  * Beside the method above, I found sklearn summary about its available measurements is pretty good: http://scikit-learn.org/stable/modules/clustering.html#clustering-performance-evaluation
    * Not only the code, but also advantage and disadvantages
    * To sum up, sklearn Silhouette Score and Calinski-Harabaz Index all work better for convex clusters than non-convex clusters. Other sklearn measurements requires ground truth, measures majorly about similarity, agreement, etc. But they don't make assumptions about the cluster structure.

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
