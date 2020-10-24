Outliers Detection and Clustering are related to each other, and in a world without groud truth data, it requires more efforts to find effective and efficient methods.

<p align="center">
  <img width="500" height="400" src="https://github.com/hanhanwu/Hanhan_Data_Science_Practice/blob/master/Outliers_and_Clustering/images/clustering_overall.PNG">
</p>


## LEARNING NOTES
* As [my data mining bible reading notes][1] recorded (Chapter 11, high demensional clustering), even it's biclustering (such as <b>MaPle</b>), which searches subsapace in both objects and features, can be time consuming because it enumerates all the subspaces. Here is another implemented example of [HiCS, LOF with description][2], [HiCS code only][3], according to the author, HiCS can solve the subspaces search in an more effective way. I think so, since MaPle is publised in 2008, HiCS is published in 2012. So deserve to try
* People have also implemneted another method to detect outliers, [LOF][4], it is densitiy based, and calculates nearest neighbours. Note that, when there is high dimensional features, PCA (linear regression model), LOF (proximity-based) can be less effective. This is the so-called The Curse of Dimensionality, when there are more dimensions, these methods can be time consuming, and outliers, random noise could make the calculation results lose meaning

### Clustering Evaluation Methods
* NMI (Normalized Mutual Information)
  ![NMI](https://github.com/hanhanwu/Hanhan_Data_Science_Practice/blob/master/Outliers_and_Clustering/NMI.png)
    * It measures the purity of Ci by calculating the largest number of common objects that cluster Ci has with all the other mutual clusters Mi. Higher NMI, the higher purity of Ci is.
* We have NMI to measure between cluster similarity, we should also measure the within ckuster purity to guarantee the ckustering quality

### Measure Clustering Quality
* Extrinsic Methods - Measure with ground truth
* Intrinsic Methods - Measure how well the clusters are seperated, without ground truth
#### Extrinsic Method - with ground truth
* Cluster homogeneity - check how pure the clusters are
* Cluster completeness - counterpart of Cluster homogeneity, if 2 projects belong to the same category, they should be in the same cluster
* Rag bag - A “rag bag” category containg objects that cannot be merged with other objects. The rag bag criterion states that putting a heterogeneous object into a pure cluster should be penalized more than putting it into a rag bag
* Small cluster preservation - The small cluster preservation criterion states that splitting a small category into pieces is more harmful than splitting a large category into pieces.
* Example - BCube, evaluates the precision and recall for every object in a clustering. The <b>precision</b> of an object indicates how many other objects in the same cluster belong to the same category as the object. The <b>recall</b> of an object reflects how many objects of the same category are assigned to the same cluster.
#### Intrinsic Method - without ground truth
* Takes advantage of similarity metrics between objects.
* Example - silhouette coefficient, valeus are between [-1, 1].  When it approaches 1, the cluster containing object o is compact and o is far away from other clusters, which is the preferable case. When its negative, it means o is closer to the objects in another cluster, bad case. Calcuate silhouette coefficient value for each object, then use the average silhouette coefficient value of all objects in the data set.
* Intrinsic methods can also be used in the elbow method to heuristically derive the number of clusters in a data set by replacing the sum of within-cluster variances.
  
### PyOD - An Outlier Detection Python Library
* My code - Basic PyOD:https://github.com/hanhanwu/Hanhan_Data_Science_Practice/blob/master/Outliers_and_Clustering/try_PyOD.ipynb
  * Test on Simulated Data: PyOD provides functions to generate simulated data which allows you to set anomalies proportion
  * <b>NOTE:</b> There is an update of plotly interactive 3D scatter plot. I stead of using the code appeared in my ipython, now better to use the code here: https://plotly.com/python/3d-scatter-plots/
  * In 2D Simulated Data Experiment
    * There are only 200 records, 200*200 meshgrid points need to calculate anomaly scores, but these algorithms performed very slow in PyOD, even running for hours without output:
      * LSCP, SOS, LOCI
  * In 3D Simulated Data Experiment
    * They all made good prediction
    * To plot those 3D images, you need to have a plotly account in order to get your username and API key
  * In Muti-dimensional (3D+) Data Experiment
    * When we have 3+ dimensions, proper amount of higher dimensions could sever for better prediction, therefore, when we are using X_train for all the classifiers here, most of them will still perform well.
    * T-SNE is good to be used in 2D, 3D plots, but converting a high dimensional data into 2D, 3D before doing the prediction, tend to make more classifiers have wrong predictions, majorly because of the missing info, and the data structure has been changed.
  * More observations can be found in the IPython
* All the algorithms it supports: https://pyod.readthedocs.io/en/latest/
  * Linear Models
    * PCA (Principle Component Analysis)
      * It's often used for feature selection, since it explains the variance-covariance structure of features through a linear function of principle components. The top k mutually independent principle components has the highest variance, and the combination of their variance equals to the variance of all the features. That's why you can choost the top k principle components as the result of dimensional reduction.
      * When it comes to anomaly detection, the basic idea here is:
        * Firstly, they use Mahalanobis metric to identify obvious observations that are significantly different from normal observations
        * They assumes outliers will bring higher variance and correlation (covariance), and trim the outliers
        * paper: https://homepages.laas.fr/owe/METROSEC/DOC/FDM03.pdf
    * MCD (Minimum Covariance Determinant)
      * Given n data points, the MCD of those data is the mean and covariance matrix based on the sample of size h (h <= n) that minimizes the determinant of the covariance matrix.
      * Effective for multivariate location and scatter (works well when there are 3+ features). When Mahalanobis distance is not robust enough to detect outliers (mask effect), MCD develops a distributional fit to Mahalanobis distances which uses a robust shape and location estimate.
    * OCSVM (One-Class Support Vector Machines)
      * One class SVM + RBF kernel function is the commonly used one
      * The hyperplane seperates the outliers and the normal data
  * Proximity-Based Models
    * LOF (Local Outlier Factor)
      * By comparing the local density of an object to the local densities of its neighbors, one can identify regions of similar density, and points that have a substantially lower density than their neighbors. These are considered to be outliers.
    * CBLOF	(Clustering-Based Local Outlier Factor)
      * It classifies the data into small clusters and large clusters. The anomaly score is then calculated based on the size of the cluster the point belongs to, as well as the distance to the nearest large cluster. With the anomaly score, it shows the degree of outlying, instead of binary outliers or not.
    * LOCI (Fast outlier detection using the local correlation integral)
      * LOCI is very effective for detecting outliers and groups of outliers. 
      * It provides an automatic, data-dictated cutoff to determine whether a point is an outlier-in contrast.
      * LOCI leads to a practically linear approximate method, aLOCI (for approximate LOCI), which provides fast highly-accurate outlier detection.
      * It provides a LOCI plot for each point which summarizes a lot of the information about the data in the area around the point, determining clusters, micro-clusters, their diameters, and their inter-cluster distances. This is a very special feature.
      * NOTE: this method doesn't need random state
    * HBOS (Histogram Based Outlier Score)
      * It is an efficient unsupervised method which assumes the feature independence and calculates the outlier score by building histograms
      * It is much faster than multivariate approaches, but at the cost of less precision. It's also ot good at detecting local outliers.
      * NOTE: this method doesn't need random state
      * How it works: https://www.researchgate.net/publication/231614824_Histogram-based_Outlier_Score_HBOS_A_fast_Unsupervised_Anomaly_Detection_Algorithm
    * KNN, AvgKNN, MedKNN
      * KNN - the distance to the kth nearest neighbor as the outlier score
      * AvgKNN - use the average distance to all the k nearest neighbors as the outlier score
      * MedKNN - use the median distance to all the k nearest neighbors as the outlier score
      * NOTE: these method doesn't need random state
  * Probabilistic Models
    * ABOD, FastABOD (Angle-Based Outlier Detection)
      * It considers the relationship between each point and its neighbor(s). It does not consider the relationships among these neighbors. The variance of its weighted cosine scores to all neighbors are viewed as the outlying score.
      * ABOD works for multi-dimensional data but time consuming. FastABOD uses KNN to approximate the results and improve the time efficiency, however this approximation will get worse when the data dimensionality increases
      * NOTE: this method doesn't need random state
      * The paper: https://imada.sdu.dk/~zimek/publications/KDD2008/KDD08-ABOD.pdf
    * SOS (Stochastic Ourlier Selection)
      * How SOS works: https://www.datascienceworkshops.com/blog/stochastic-outlier-selection/
        * It uses "affinity" (similarity), the idea used by tSNE (tSNE uses it to preserve local structure of the high dimensional data), SOS uses this idea to detect outliers. SOS computes an affinity matrix A, a binding probability matrix B, and finally, the outlier probability vector Φ.
        * You can also find the AUC comparison between SOS and other outlier detection methods. SOS is more robust to data perturbations (noisy data) and varying densities.
      * NOTE: this method doesn't need random state
  * Outlier Ensembles
    * IForest (Isolation Forest)
      * The algorithm isolates each point in the data and splits them into outliers or inliers. Data partitioning is done using a set of trees. It counts the number of times a point can be isolated from other points, the count is used in outlying score.
      * The .gif shows how IForest work https://quantdare.com/isolation-forest-algorithm/
      * But this algorithm could suffer from failure due to irrelevant data dimensions
    * Feature Bagging
      * A feature bagging detector fits a number of base detectors on various sub-samples of the dataset. It uses averaging or other combination methods to improve the prediction accuracy. By default, Local Outlier Factor (LOF) is used as the base estimator. However, any estimator could be used as the base estimator, such as kNN and ABOD.
      * Feature bagging first constructs n sub-samples by randomly selecting a subset of features. This brings out the diversity of base estimators. Finally, the prediction score is generated by averaging or taking the maximum of all base detectors.
    * [2019] LSCP (Locally Selective Combination of Parallel Outlier Ensembles)
      * The selection of base estimator will affect the model accuracy and stability. LSCP is designed to addresses the issue by defining a local region around a test instance using the consensus of its nearest neighbors in randomly selected feature subspaces. The top-performing base detectors in this local region are selected and combined as the model’s final output.
      * It like the pseudo ground truth generation idea used here for estimator selection.
      * The paper: https://static1.squarespace.com/static/56368331e4b08347cb5555e1/t/5c47d75bb91c915700195753/1548212060246/SCP_draft.pdf
    * [2018] XGBOD (Extreme Gradient Boosting Outlier Detection) - Supervised
      * It uses mutiple unsupervised models to cluster the data and generates the transformed outlier score (TOS), the TOS scores are combined with original features as new feature set. After feature selection, it uses XGBoost to predict the final outlier result.
      * The paper, it's ferr to download: https://www.researchgate.net/publication/328399475_XGBOD_Improving_Supervised_Outlier_Detection_with_Unsupervised_Representation_Learning
  * Neural Netwrok Models
    * Fully connected AutoEncoder, Single-Objective Generative Adversarial Active Learning, Multiple-Objective Generative Adversarial Active Learning
* IPython Notebook - compare all algorithms: https://github.com/yzhao062/pyod#api-cheatsheet--reference
  

## PRACTICAL CODE
* Previous Experience Notes: https://github.com/hanhanwu/Hanhan_Data_Science_Resources/edit/master/Experiences.md
  * Suggested data preprocessing for clustering
  * 4 rounds clustering when using clustering methods

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
    * Higher the score, the data is better clustered. So when choosing the optimal k, choose the one give the highest silhouette score.
      * If Elbow will return SSE, we can use both Silhouette Score and Elbow, choose k that has higher Silhouette Score and lower SSE.
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
    
## All About Algorithms
### K-Means vs K-Medoids vs K-Means++
* K-Means
  * It randomly selects k objects as the initial center
  * It calculates the mean value as the center of each group. Therefore it's easy to be affected by outliers
* K-Medoids
  * In order to overcome the outlier issues in K-Means, it choses existing objects (randomly selected) to replace the previous centroid in order to minimize within cluster distances
* K-Means ++
  * Similar to K-Medoids, but instead of selected the next centroid randomly, it uses a weighted probability distribution where a point x is chosen with probability proportional to D(x)2
    * D(x)2 - Square distance to current centroid x
    * For example it can choose the next centroid as the one whose D(x)2 is farthest from current centroid x
    
### Gaussian Mixture Models (GMMs)
* "Gaussian Mixture Models (GMMs) assume that there are a certain number of Gaussian distributions, and each of these distributions represent a cluster. Hence, a Gaussian Mixture Model tends to group the data points belonging to a single distribution together."
  * In each dimension, the PDF (probability density function) is a gaussian distribution.
  * The PDF for GMM is similar to gaussian distribution formula, but having `x` and `μ` as vectors of length `d`, and `Σ` would be a `d * d` covariance matrix.
* EM in GMMs
  * Mean and variance are assigned through Expectation-Maximization (EM), a statistical algorithm used for finding the right model parameters.
  * EM is often used when there is misssing variable, in unsupervised learning case, it's the label. We call any missing variable as "latent variable". EM tries to use the existing data to determine the optimum values for latent variables and then finds the model parameters.
    * E-step: The available data is used to estimate (guess) the values of the missing variables
    * M-step: Based on the estimated values generated in the E-step, the complete data is used to update the parameters
  * In GMMs, 
    * E-step: calculates the probability that a record belongs to each cluster/distribution
    * M-step: update `Π`, `μ` and `Σ` values
      * `Π` is density, `Π = number of records in a cluster/total number of records`
    * E-M process is repeated to [maximize log likelihood][6]. 
  * GMMs vs k-kmeans
    * GMMs considers both mean and variance, while k-means only cares mean.
    * As we can see from [this example][7], when a model only checks distance, might not cluster the data in the right way. GMMs might come to help.
* [Reference][5]

### Density Based CLustering
#### [DBSCAN][8]
* Partitioning and hierarchical methods are designed to find spherical-shaped clusters. They have difficulty finding clusters of arbitrary shape such as the “S” shape and oval clusters. Given such data, they would likely inaccurately identify convex regions, where noise or outliers are included in the clusters. In such cases, density based clustering methods come to help.
* 2 params
  * `Epsilon` is the radius of the circle to be created around each data point to check the density
  * `minPoints` is the minimum number of data points required inside that circle for that data point to be classified as a Core point.
    * suggest to have `minPoints>=Dimensions+1` to avoid each single feature occupies a circle
* Core vs Border vs Noise
  * A data point is a "Core" point if the circle around it contains at least ‘minPoints’ number of points, including itself
  * If the number of points is less than minPoints but larger than 1 inside of the circle, including itself, then it is classified as "Border" Point
  * Data points with no point other than itself present inside the circle are considered as "Noise" point
* Reachability and Connectivity
  * Directly density-reachable: Y is the core point, X is directly reachable from Y and they are in the same circle. Vice versa might not be true.
  * Density-reachable: Y is the core point, and it can reach to X through its directly density-reachable points.
  * Density-connected: O is a core point, both X and Y are density-reachable from O, then X and Y are density-connected
  * Reachability is not symmetric (may not vice versa), connectivity is symmetric
  * How to use reachability and connectivity
    * If a point is density-reachable from some point of the cluster, it is part of the cluster as well.
      * Note, cluster is different from circles
    * All points within the cluster are mutually density-connected.
* Disadvantage
  * Just like many other clustering methods, DBSCAN requires the user to select the params for the discovery of the clusters. But this can be hard to determine, especially in real world with high dimensional data.
    * In real world, high dimensional data also tend to have highly skewed distribution, therefore their intrinsic clustering structure may not be well characterized by a single set of global density parameters.
#### [OPTICS][9]
* It can overcome the disadvantage of DBSCAN, it doesn't need the user to provide specific density threshold.
#### DENCLUE
* It's a clustering method based on a set of density distribu- tion functions.
* In DBSCAN and OPTICS, density is calculated by counting the number of objects in a neighborhood defined by a radius parameter. Such density estimates can be highly sensitive to the radius value used. 
* To overcome this problem, kernel density estimation can be used, which is a nonparametric density estimation approach from statistics. DENCLUE uses a Gaussian kernel to estimate density based on the given set of objects
to be clustered.


## IDEAS SPARK
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
[5]:https://www.analyticsvidhya.com/blog/2019/10/gaussian-mixture-models-clustering/?utm_source=feedburner&utm_medium=email&utm_campaign=Feed%3A+AnalyticsVidhya+%28Analytics+Vidhya%29
[6]: https://www.analyticsvidhya.com/blog/2018/07/introductory-guide-maximum-likelihood-estimation-case-study-r/
[7]:https://github.com/hanhanwu/Hanhan_Data_Science_Practice/blob/master/Outliers_and_Clustering/GMMs_vs_kmeans.ipynb
[8]:https://www.analyticsvidhya.com/blog/2020/09/how-dbscan-clustering-works/?utm_source=feedburner&utm_medium=email&utm_campaign=Feed%3A+AnalyticsVidhya+%28Analytics+Vidhya%29
[9]:https://scikit-learn.org/stable/modules/generated/sklearn.cluster.OPTICS.html
