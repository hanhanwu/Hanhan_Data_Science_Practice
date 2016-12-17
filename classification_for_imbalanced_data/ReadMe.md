
Recently, I did many experiments on classification with imbalanced small dataset. Although I still have many experiments in plan before Dec. 31 2016, it's time to record my summary here.


***********************************************************************

<b>Overview</b>

* The dataset I used here is very small, only around 6000+ records and it has severe data imbalance problem. Because of privacy issue, I cannot share anything detail about the data. But I need to write donw the summary to help my future data science work, since by doing these experiments, I am learning a lot.
* In my current experiments, I have tried different data preprocessing methods/libraries, along with my favorite classifcation algorithms. The results were quite different from what I have learned from online tutorials but it is very interesting.
* Here I am storing my sample code for data explore process, data preprocessing, feature selection wih different libraries and model training & evaluaion.


***********************************************************************

<b>Data Explore</b>

* There are 200+ features, but I did data exploration by checking feature by feature, at the same time, I was recording potential features that need to deal with outliers, or should using binning to reduce levels, or should using one-hot encoding, or the feature is almost 0 variance and can be removed.
* About data exploration, I really like this post: https://www.analyticsvidhya.com/blog/2016/01/guide-data-exploration/
* However, when working on real world data, even if we do all the steps mentioned in the above post, we may not get the optimal results, and sometimes, at least in my experiments, some methods brounght much worse results.
