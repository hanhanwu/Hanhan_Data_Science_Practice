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
* A simple linear generative model with Gaussian latent variables.
##### PCA (Principle Component Analysis)
* A "principle component" is a linear combination of the original features.
* We choose top n principle components that will explain most of the variance. From the top 1 principle component to top n principle component, the contribution to variance from each of them is decreasing. 
  * This is why Explained Variance Ratio curve is decreasing, while Cumulative Explained Variance Ratio curve is increasing.[My PCA code and plots][6]
##### Factor Analysis vs PCA
* There is an experiment showing that with homoscedastic noise both FA and PCA succeds. However PCA fails and overestimates the rank when heteroscedastic noise is present.
  * Homoscedastic noise - noise variance is the same for each feature
  * Heteroscedastic noise - noise variance is different for each feature
  * I think this is because PCA chooses top principle compoents that explain most of the variance, then the variance between noise could be misleading to PCA.
##### ICA (Independent Component Analysis)
* A "component" here is also a linear combination of the original features.
* But the components in ICA are independent, while the components in PCA are uncorrelated.
  * Maximizing the kurtosis will make the distribution non-gaussian and hence we will get independent components.
  * "One common measure of shape is called the kurtosis. As skewness involves the third moment of the distribution, kurtosis involves the fourth moment. The outliers in a sample, therefore, have even more effect on the kurtosis than they do on the skewness  and in a symmetric distribution both tails increase the kurtosis, unlike skewness where they offset each other."
##### SVD (Singular Value Decomposition)
* It uses Eigenvalues and Eigenvectors to decompose original variables into constituent matrices for dimensional reduction.
* Majorly used to reduce redundant features.
#### Non-Linear Methods
##### t-SNE
* It's trying to retain both local and global structure of the data at the same time, by doing these:
  * Local approaches :  They maps nearby points on the manifold to nearby points in the low dimensional representation.
  * Global approaches : They attempt to preserve geometry at all scales, i.e. mapping nearby points on manifold to nearby points in low dimensional representation as well as far away points to far away points.
##### UMAP (Uniform Manifold Approximation and Projection)
* UMAP is fast. It can handle large datasets and high dimensional data without too much difficulty, scaling beyond what most t-SNE packages can manage.
* UMAP scales well in embedding dimension -- it isn't just for visualisation! You can use UMAP as a general purpose dimension reduction technique as a preliminary step to other machine learning tasks. 
* UMAP often performs better at preserving aspects of global structure of the data than t-SNE. This means that it can often provide a better "big picture" view of your data as well as preserving local neighbor relations.
* UMAP supports a wide variety of distance functions, including non-metric distance functions such as cosine distance and correlation distance. You can finally embed word vectors properly using cosine distance!
* UMAP GitHub: https://github.com/lmcinnes/umap
##### ISOMAP
* Used when the data is strongly non-linear
#### References
* [Python Dimensional Reduction Methods][7]
  * Some methods sklearn has built-in, no need that complex
  * I don't really think the visualization here is a good way to show how well these methods work. Just a reference here, still has something to learn from.

[1]:https://github.com/hanhanwu/Hanhan_Data_Science_Resources
[2]:https://github.com/hanhanwu/Hanhan_Data_Science_Resources/blob/master/Experiences.md
[3]:https://github.com/hanhanwu/Hanhan_Data_Science_Resources2
[4]:https://github.com/hanhanwu/Hanhan_Data_Science_Practice/tree/master/classification_for_imbalanced_data
[5]:https://github.com/hanhanwu/Hanhan_Data_Science_Practice
[6]:https://github.com/hanhanwu/Hanhan_Data_Science_Practice/blob/master/make_sense_dimension_reduction.ipynb
[7]:https://www.analyticsvidhya.com/blog/2018/08/dimensionality-reduction-techniques-python/?utm_source=feedburner&utm_medium=email&utm_campaign=Feed%3A+AnalyticsVidhya+%28Analytics+Vidhya%29
