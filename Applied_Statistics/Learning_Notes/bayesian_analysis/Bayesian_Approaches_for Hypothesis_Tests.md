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
    * For parameters p1, p2, their marginal distribtions could be the same but conjoint distribution could be positive or negative as what's shown above.
    * Marjinal distribution (the histogram) cannot really tell their relationship, even if the distributions could have large overlap.
  * To check the difference, bettwe to check whether 0 is within 95% HDI of `p1-p2` difference distribution.
    * When 0 is outside 95% HDI, means the difference is credible, otherwise it's not.
    * For positive correlation, the difference distribution can be narrower. For negative correlation, the difference distribution can be wider.
## Approach 2
