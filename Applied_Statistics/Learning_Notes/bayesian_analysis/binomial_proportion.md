# Binomial Proportion via Mathematical Analysis

## The Likelihood Function - Bernoulli Distribution
* `p(y|θ) = power(θ, y) * power(1-θ, 1-y)`
  * This is the likelihood function of θ, or "Bernoulli likelihood function".
  * y has 2 nominal values (categorical without order), such as 0, 1.
  * If θ is the probability of y=1, then 1-θ is the probability of 1-y.
* Bernoulli distribution is a DISCRETE distribution over 2 values of y. The likelihood function specifies the probability at each value of θ but it's not a probability distribution.

## The Probability Density - Beta Distribution
* `p(θ|a,b) = beta(θ;a,b) = power(θ, a-1) * power(1-θ, b-1) / B(a,b)`
  * B(a, b) is simply a normalizing constant that ensures that the area under the beta density integrates to 1.0.
  * a,b a the params of beta distribution, they must be positive values.
    * a+b is N, total number of trails, such as `a` number of y=1, `b` number of y=0.
  * When a,b, get bigger together, beta distribution gets narrower.
* Mean of beta distribution
  * `μ=a/(a+b)`
* Standard Deviation of beta distribution
  * `std = sqrt(μ*(1-μ)/(a+b+1))`
* Exmaple of Beta Distributions
<p align="center">
<img src="https://github.com/hanhanwu/Hanhan_Data_Science_Practice/blob/master/Applied_Statistics/Learning_Notes/images/beta_distribution_examples.PNG" width="400" height="400" />
</p>

## 3 Inferential Goals
### Estimating the Binomial Proportion
* The posterior distribution indicates which values of θ are more credible than others.
  * Such as using 95%, 89% HDI. Any point inside of HDI has higher believability than points outside of the HDI.
  * 95% HDI is fairly wide when the prior is uncertain, but narrower when the prior is more certain.
### Predicting Data
* The predicted probability of a datum value y is determined by averaging that value’s probability across all possible parameter values,
weighted by the belief in the parameter values.
<p align="center">
<img src="https://github.com/hanhanwu/Hanhan_Data_Science_Practice/blob/master/Applied_Statistics/Learning_Notes/images/ptheta.PNG" width="200" height="50" />
</p>
* For example, the predicted probability of y=1 is the mean of posterior distribution over θ, `(z+a)/(N+a+b)`.
  * `z` is the number of y=1 among N.
### Model Comparison
* Equal prior probabilities
  * Then just to compare the bayes factor (ratio of evidence).
* Unequal prior probabilities
  * Then the priors must be factored into posteriors
  * If we have strong prior beliefs (such as uniform distribution) in one model, it takes more evidence for the other model to overcome the prior belief.
* Relative Believability
  * The winning model only has a relative believability, not absolute believability.
* Posterior Predictive Check - Assess whether the winning model can actually account for the data
  * To simulate data sample from the winning model and see if the simulated data look like the actual data.
    * Randomly generate θ from the posterior distribution of the winning model.
    * Using θ, generate a sample of coin flips.
    * Count the number of heads in the sample as a summary of the sample.
    * Determine whether the number of heads in a typical simulated sample is close to the number of heads in the actual sample.
  * Chapter 5, page 82, there is R code
  
  ## Summary - How to do Bayesian Inference with Binomial Proportion
  * Data Simulation
    * 2 nomial values
    * The 2 values must come up randomly, independently across observations with a single and fixed probability.
  * Establish the prior belief with a beta distribution `beta(θ; a,b)`
    * Decide what you think is the most probable value for θ, call it `m`
    * Decide how strong you believe in `m` by considering how many new data points (flips of the coin) it would take to sway you away from the prior belief. Call it as `n`.
    * `a = mn`
    * `b = (1-m)n`
    * If you have m and var or std
      * `a = m(m(1-m)/var - 1)`
      * `b = (1-m)(m(1-m)/var - 1)`
  * Check beta distribution, and data samples after getting a,b.
    * Using the R code in Chapter 5, page 77
  * Determine the posterior distribution of beliefs regarding values of θ
  * Make inferences from the posterior, depending on the goal
    * Goal to estimate θ
      * Check 95%, 89% HDI in posterior distribution
    * Goal to predict new data
      * predicted probability of “heads” on the next flip is the mean of the posterior, which is `(z + a)/(N + a + b)`
    * Goal to compare models
      * Bayes factor, or bayes factor with prior beliefs
      * Then use posterior predictive check to get a sense of whether the better model actually mimics the data well
