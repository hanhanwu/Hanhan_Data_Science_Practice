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
  * When a,b, get bigger together, beta distribution gets narrower.
* Mean of beta distribution
  * `μ=a/(a+b)`
* Standard Deviation of beta distribution
  * `std = sqrt(μ*(1-μ)/(a+b+1))`
  
* Exmaple of Beta Distributions
<p align="center">
<img src="https://github.com/hanhanwu/Hanhan_Data_Science_Practice/blob/master/Applied_Statistics/Learning_Notes/images/beta_distribution_examples.PNG" width="400" height="400" />
</p>
