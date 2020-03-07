# Bayesian Approaches to Hypothesis Tests

## Approach 1 - The estimation (Single Prior) Approach
* <b>A parameter value</b> is NOT credible if it lies outside of the 95% HDI of the posterior distribution of that parameter. Otherwise when it's within the 95% HDI, it's among the credible values.
  * It address the credibility by considering the posterior estimate of the parameter, derived from a single prior distribution.
* Examples
  * Whether a parameter value is credible
  * Whether a difference of parameters is credible
* Differences of correlated parameters
  * Conjoint Distribution Examples
    * Left is positive correlation, right is negative correlation
  <p align="left">
  <img width="200" height="200" src="https://github.com/hanhanwu/Hanhan_Data_Science_Practice/blob/master/Applied_Statistics/images/conjoint_dist_pos.png">
   <img width="200" height="200" src="https://github.com/hanhanwu/Hanhan_Data_Science_Practice/blob/master/Applied_Statistics/images/conjoint_dist_neg.png">
  </p>

  * <b>The marginal distributions of single parameters do not indicate the relationship between them.</b>
    * Marginal distribution is the probability distribution of the dataset, namely it's the histogram of that distribution.
    * For parameters p1, p2, their marginal distribtions could be the same but conjoint distribution could be positive or negative as what's shown above.
    * The overlapped space of marginal distributions of 2 datasets, can NOT really tell their relationship/difference.
  * To check the difference, better to check whether 0 is within 95% HDI of `p1-p2` difference distribution.
    * The images below are the distribution for the differences, `p1-p2`. The left one is for the above positive correlation, while the right is the one for the above neagtive correlation.
    * For positive correlation, the difference distribution can be narrower. For negative correlation, the difference distribution can be wider.
    * In the left distribution, 0 is outside of 95% HDI, because when one value increases the other also increases, therefore we can see almost all the difference points dropped on one side of the equality line. However when it's negative correlation, the difference can be smaller or larger, that's also why the distribution is wider.
    * <b>"Marginal distributions of single parameters do not indicate their relationship, since the parameters can have positive or negative correlation. Therefore the difference between parameters should be explicitly examined."</b>
  <p align="left">
  <img width="200" height="200" src="https://github.com/hanhanwu/Hanhan_Data_Science_Practice/blob/master/Applied_Statistics/images/post_diff_dist.PNG">
   <img width="200" height="200" src="https://github.com/hanhanwu/Hanhan_Data_Science_Practice/blob/master/Applied_Statistics/images/neg_diff_dist.PNG">
  </p>
    
## Approach 2
