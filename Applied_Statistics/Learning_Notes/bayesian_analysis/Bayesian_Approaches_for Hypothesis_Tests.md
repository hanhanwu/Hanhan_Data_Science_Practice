# Bayesian Approaches to Hypothesis Tests

## Approach 1 - The estimation (Single Prior) Approach
* <b>A parameter value</b> is NOT credible if it lies outside of the 95% HDI of the posterior distribution of that parameter. Otherwise when it's within the 95% HDI, it's among the credible values.
  * It address the credibility by considering the posterior estimate of the parameter, derived from a single prior distribution.
* Examples
  * Whether a parameter value is credible
    * The null value here is the parameter value, and needs to decide whether to reject this null value.
  * Whether a difference of parameters is credible
    * The null value here is the difference value, and needs to decide whether to reject this null value.
  * Region of Practical Equivalance (ROPE)
    * It's a small range of values
    * A parameter value is NOT credible or rejected once the entire ROPE lies outside the 95% HDI of the posterior distribution of that parameter.
    * A parameter value is accepted for practical purpose if the value's ROPE completely CONTAINS the 95% HDI of the posterior of that parameter.
      * As the sample size grows, HDI will become narrower and closer to the true value. So if the ROPE around the null value is true, finally we will accept the null value when the sample size grows large enough.
    * The broader goal of Bayesian Analysis is conveying an informative summary of the posterior, and where the value of interest falls within that posterior. Reporting the limits of an HDI region is more informative than reporting the declaration of a reject/accept decision.
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
* <b>"Bayesian model comparison is only useful when both models are genuinely viable and equivalently informed. If one or both models has little prior believability, then the Bayes factor and relative posterior believabilities are of little use."</b>
  * Both models should have prior viability, not just one.
* Instead of focusing on the magnitude of the parameter, it suggests to focus on deciding which of 2 hypothetical priors is least unbelievable.
  * One hypothetical prior expresses the idea that the parameter value is exactly the null value.
  * The alternative hypothetical prior expresses the idea that the parameter could be anything but the null value.
* To compare the models:
  * `P(M_alt|D)/P(M_null|D) = (P(D|M_alt) * P(M_alt))/ (P(D|M_null) * P(M_null))`
    * The prior beliefs in each model are typically assumed to be equal. `P(M_null) = P(M_alt)`
    * So we majorly use <b>Byes Factor (BF)</b>, `Bayes Factor = (P(D|M_alt)/(P(D|M_null)`
      * When P(D|M_alt) is close to P(D|M_null), BF is just slightly favors the alternative or the null, so either model remains reasonably credible.
      * If the Bayes factor had turned out to be more extreme, we might decide to declare one or the other prior to be less unbelievable than the other prior.
      
## Credible Intervals
* Different Types of Credible Intervals
  * HDI (High Density Interval) - all points within the interval have a higher probability density than points outside the interval.
  * ETI (Equal-tailed Interval) - the ETI is equal-tailed. This means that a 90% interval has 5% of the distribution on either side of its limits. It indicates the 5th percentile and the 95th percentile
  * In symmetric distributions, the two methods of computing credible intervals, the ETI and the HDI, return similar results. However,
    * HDI is better at summarizing the credible valuesin distributions, because it's possible that parameter values in the ETI have lower credibility than parameter values outside of the ETI.
    * But when the distribution has been transformed (such as log transformation), ETI is better than HDI, since the lower and higher bounds of the transformed distribution in EIT will correspond to the transformed lower and higher bounds of the original distribution.
* 95% vs. 89%
  * We were educated to use 95% credible interval, but some suggested that 95% might not be the most apppropriate for Bayesian posterior distributions, potentially lacking stability if not enough posterior samples are drawn.
  * Thus default credible interval is 89%,in order to have a more stable choice.
* The Support Interval (SI)
  * Unlike the HDI and the ETI, which look at the posterior distribution, the Support Interval (SI) provides information regarding <b>the change in the credability of values from the prior to the posterior</b>
  * It indicates which values of a parameter are have gained support by the observed data, or SI is an interval that contains only those values whose credibility is not decreased by observing the data
  * [More about the support interval][3]

## Reference
* [Doing Bayesian Data Analysis][1]
* [Credible Intervals in R][2]

[1]:[2]:https://www.amazon.com/Doing-Bayesian-Data-Analysis-Tutorial/dp/0123814855/ref=cm_cr_arp_d_product_top?ie=UTF8
[2]:https://easystats.github.io/bayestestR/articles/credible_interval.html
[3]:https://link.springer.com/article/10.1007/s10670-019-00209-z
