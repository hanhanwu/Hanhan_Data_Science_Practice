# Null Hypothesis Significance Testing (NHST) vs Bayesian Analysis

* In NHST, the goal of <b>inference</b> is to decide whether a particular value `𝜃` of a model parameter can be rejected.
* A good experiment is founded on the principle that the data is insulated from the experimenter's intentions.

## About p-value
* We want to know the probability of getting the actual outcome or an outcome more extreme relevant to what we expect. This is total probability is "p-value".
  * When p-value is less than a critical amount, we reject the null hypothesis.
  * However, when we are rejecting the null hypothesis, we still have no particular degree of disbelief in it, we don't have any particular disbelief in any other hypothesis neither. <b>All what we know is, the actual observation lies in an extreme end of the space of possibility</b>, if the intended experiment were repeated.
  
## NHST vs Bayesian Analysis
### Experimenter's Intention
* NHST analysis depends on the intentions of the experimenter, and these intentions define the space of all possible (unobserved) data.
  * For example, an experimenter intends to fix N, or intends to fix the number of heads of fliping coins.
* However the dependence on experimenter's intertions is conflicting with the opposite assumption that the experimenter's intentions have no effect on the observed data.
* Bayesian analysis does not depend on the space of possible unobserved data, it only operates on the actual observed data.
### Confidence Interval vs HDI (highest density interval)
* In NHST, confidence interval tells us the probability of extreme unobserved values that we might get if we repeat the experiment based on experimenter's intention. But the confidence interval doesn't tell much about the believability of any value `𝜃`.
* In bayesian analysis, there is HDI, which tells minimal level of posterior believability, using P(𝜃|D), given the data, what's the probability of `𝜃`, which is exactly what we are looking for.
  * 95% HDI of values of `𝜃` indicates the total probability of all such `𝜃` values is 95%
* HDI does not depend on experimenter's intention. By contrast, NHST confidence interval tells us about probabilities of data relative to what might have been if we replicated the experimenter's intentions.
* HDI is responsive to analyst's prior beliefs, bayesian analysis indicates how much new data should alter our beliefs and the priori beliefs are overt and public decided. In NHST, the accumulated prior is ignored.
### Differences in Comparing Various Groups
* Bayesian posterior directly tells us the believability of the magnitudes of differences.
* NHST tells us about whether the difference is extreme in a sspace of possibilities determined by experimenter's intentions.

## Benefits of Sampling Distribution
* Sampling distributions tell us the probabilities of possible data if we run an intended experiment given a particular hypothesis, rather than the believabilities of possible hypothesis given a particular set of data.
* The posterior distribution only tells us which parameter is relatively better than others, but does 't tell whether the best parameter is really a good one (a good description of the data).
* In order to check whether the best parameter value is in fact good, we can use <b>posterior predictive check</b>. If the posterior parameter values are real good descriptions of the data, then the predicted data from the model should look like the real data.
  * The use of the posterior predictive check is similar to null hypothesis testing. We start with a hypothesis and generate simulated data as if we were repeating the intended experiment over and over again. Then check whether the actual data are typical or atypical in the space of simulated data.

## Reference
* [Doing Bayesian Data Analysis][1]

[1]:https://www.amazon.com/Doing-Bayesian-Data-Analysis-Tutorial/dp/0123814855/ref=cm_cr_arp_d_product_top?ie=UTF8
