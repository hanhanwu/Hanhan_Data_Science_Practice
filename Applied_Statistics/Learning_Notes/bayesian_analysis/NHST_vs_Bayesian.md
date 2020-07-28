# Null Hypothesis Significance Testing (NHST) vs Bayesian Analysis

* In NHST, the goal of <b>inference</b> is to decide whether a particular value `𝜃` of a model parameter can be rejected.
* A good experiment is founded on the principle that the data is insulated from the experimenter's intentions.

## About p-value
* We want to know the probability of getting the actual outcome or an outcome more extreme relevant to what we expect. This is total probability is "p-value".
  * When p-value is less than a critical amount, we reject the null hypothesis.
  * However, when we are rejecting the null hypothesis, we still have no particular degree of disbelief in it, we don't have any particular disbelief in any other hypothesis neither. <b>All what we know is, the actual observation lies in an extreme end of the space of possibility</b>, if the intended experiment were repeated.
  
## NHST vs Bayesian Analysis
* “Bayesian statistics is a mathematical procedure that applies probabilities to statistical problems. It provides people the tools to update their beliefs in the evidence of new data.”

### Experimenter's Intention
* NHST analysis depends on the intentions of the experimenter, and these intentions define the space of all possible (unobserved) data.
  * p-values measured against a sample (fixed size) statistic with some stopping intention changes with the change in intention and sample size.
    * For example, if two persons work on the same data and have different stopping intention, they may get two different  p-values for the same data, which is undesirable.
* Bayesian analysis does not depend on the space of possible unobserved data, it only operates on the actual observed data.
### Confidence Interval vs HDI (highest density interval)
* In NHST
  * Confidence Interval (C.I) is like p-value depends heavily on the sample size, which is affected by the stopping intention.
  * Confidence Intervals (C.I) are not probability distributions therefore they do not provide the most probable value for a parameter and the most probable values.
* In bayesian analysis, there is HDI, which tells minimal level of posterior believability, using P(𝜃|D), given the data, what's the probability of `𝜃`, which is exactly what we are looking for.
  * 95% HDI of values of `𝜃` indicates the total probability of all such `𝜃` values is 95%
* HDI does not depend on experimenter's intention. By contrast, NHST confidence interval tells us about probabilities of data relative to what might have been if we replicated the experimenter's intentions.
* HDI is responsive to analyst's prior beliefs, bayesian analysis indicates how much new data should alter our beliefs and the priori beliefs are overt and public decided. In NHST, the accumulated prior is ignored.
### Differences in Comparing Various Groups
* Bayesian posterior directly tells us the believability of the magnitudes of differences.
* NHST tells us about whether the difference is extreme in a space of possibilities determined by experimenter's intentions.

## Example of using Bayesian Analysis - Test for Significance 🌺
* [A simple example][2]
* There are H0, H1 too
* Method 1 - Using `Bayesian Factor` instead of p-value or confidence interval
  * To calculate the Posterior belief, you need to know:
    * The distribution type, most of the cases are binomial distribution
    * Get mean and std of the distribution and therefore get the parameters α， β
    * With `z, N, α， β` you will be able to calculate the posterior belief, which will be used in the bayesian factor
  * To reject a null hypothesis, a BF <1/10 is preferred
* Method 2 - Using HDI
  * Within 95% or 89% HDI, then it's significant
  * After getting more dataset, HDI is becoming narrower, so the concern of the changing of HDI should not bother


## Benefits of Sampling Distribution
* Sampling distributions tell us the probabilities of possible data if we run an intended experiment given a particular hypothesis, rather than the believabilities of possible hypothesis given a particular set of data.
* The posterior distribution only tells us which parameter is relatively better than others, but does 't tell whether the best parameter is really a good one (a good description of the data).
* In order to check whether the best parameter value is in fact good, we can use <b>posterior predictive check</b>. If the posterior parameter values are real good descriptions of the data, then the predicted data from the model should look like the real data.
  * The use of the posterior predictive check is similar to null hypothesis testing. We start with a hypothesis and generate simulated data as if we were repeating the intended experiment over and over again. Then check whether the actual data are typical or atypical in the space of simulated data.

## Reference
* [Doing Bayesian Data Analysis][1]
* [Bayesian Analysis example][2]

[1]:https://www.amazon.com/Doing-Bayesian-Data-Analysis-Tutorial/dp/0123814855/ref=cm_cr_arp_d_product_top?ie=UTF8
[2]:https://www.analyticsvidhya.caom/blog/2016/06/bayesian-statistics-beginners-simple-english/
