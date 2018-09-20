# Data Preprocessing
Finally decided to put all the future work here.

## Previous Work
* [Hanhan's Data Science Resources][1]
* [Hanhan's Experience Notes][2]
* [Hanhan's Data Science Resources 2][3]
* [Hanhan's Notes From An Industry Delivered Project 2016][4]
* [Hanhan's Data Science Practice][5]

## More

### Dimensional Reduction
#### Linear Methods
##### Factor Analysis
* It puts features into groups, features in a group have strong correlation with each other, but the between-group correlation should be lower. Each group is a "factor". So, by converting features into factors, the dimension reduced.
##### PCA (Principle Component Analysis)
* A "principle component" is a linear combination of the original features.
* We choose top n principle components that will explain most of the variance. From the top 1 principle component to top n principle component, the contribution to variance from each of them is decreasing. 
  * This is why Explained Variance Ratio curve is decreasing, while Cumulative Explained Variance Ratio curve is increasing.[My PCA code and plots][6]
##### Factor Analysis vs PCA
* There is an experiment showing that with homoscedastic noise both FA and PCA succeds. However PCA fails and overestimates the rank when heteroscedastic noise is present.
  * Homoscedastic noise - noise variance is the same for each feature
  * Heteroscedastic noise - noise variance is different for each feature
  * I think this is because PCA chooses top principle compoents that explain most of the variance, then the variance between noise could be misleading to PCA.
##### SVD (Singular Value Decomposition)
* It uses Eigenvalues and Eigenvectors to decompose original variables into constituent matrices for dimensional reduction.
* Majorly used to reduce redundant features.


[1]:https://github.com/hanhanwu/Hanhan_Data_Science_Resources
[2]:https://github.com/hanhanwu/Hanhan_Data_Science_Resources/blob/master/Experiences.md
[3]:https://github.com/hanhanwu/Hanhan_Data_Science_Resources2
[4]:https://github.com/hanhanwu/Hanhan_Data_Science_Practice/tree/master/classification_for_imbalanced_data
[5]:https://github.com/hanhanwu/Hanhan_Data_Science_Practice
[6]:https://github.com/hanhanwu/Hanhan_Data_Science_Practice/blob/master/make_sense_dimension_reduction.ipynb
