Outliers Detection and Clustering are related to each other, and in a world without groud truth data, it requires more efforts to find effective and efficient methods.


*********************************************************************************

LEARNING NOTES

* As [my data mining bible reading notes][1] recorded (Chapter 11, high demensional clustering), even it's biclustering (such as <b>MaPle</b>), which searches subsapace in both objects and features, can be time consuming because it enumerates all the subspaces. Here is another implemented example of [HiCS, LOF with description][2], [HiCS code only][3], according to the author, HiCS can solve the subspaces search in an more effective way. I think so, since MaPle is publised in 2008, HiCS is published in 2012. So deserve to try
* People have also implemneted another method to detect outliers, [LOF][4], it is densitiy based, and calculates nearest neighbours. Note that, when there is high dimensional features, PCA (linear regression model), LOF (proximity-based) can be less effective. This is the so-called The Curse of Dimensionality, when there are more dimensions, these methods can be time consuming, and outliers, random noise could make the calculation results lose meaning


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
