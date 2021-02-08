# Bayesian Analysis

More details about Bayesian Analysis


## Reference 🌟🌟
* [Doing Bayesian Data Analysis][2] - It's a book

## Bayesian Inference


## Bayesian Theorem vs Conditional Probability
* Conditional Probability
  * `P(A|B) = P(A and B)/P(B)`
  * A conditional probability is an expression of how probable one event is, given that some other event occurred.
* Bayesian Theorem
  * `P(A|B) = P(B|A)*P(A)/P(B)`
  * It came from the axiom of conditional probability, and it centerns on relating different conditional probabilities.
    * P(A|B) = P(A and B) / P(B)
    * P(B|A) = P(A and B) / P(A)
    * So P(A|B) * P(B) = P(B|A) * P(A) ==> P(A|B) = P(B|A)*P(A)/P(B)
  * Posterior probability = likihood ratio * priori probability
    * P(A) is "priori probability"
      * Conjugate Priors: It occurs when the final posterior distribution belongs to the family of similar probability density functions as the prior belief but with new parameter values which have been updated to reflect new evidence/ information. Examples Beta-Binomial, Gamma -Poisson or Normal-Normal.
      * Non-conjugate Priors: When personal belief cannot be expressed in terms of a suitable conjugate prior and for those cases simulation tools are applied to approximate the posterior distribution. An example can be Gibbs sampler.
      * Uninformative prior: Another approach is to minimize the amount of information that goes into the prior function to reduce the bias. This is an attempt to have the data have maximum influence on the posterior. But the result will be similar to frequentist approach.
    * P(B|A) is "posterior probability"
    * P(B|A)/P(B) is "likelihood ratio"
  * Bayes Theorem for continuous values
    * The y in the dividen is a specific fixed value, whereas the y in the divisor is a variable that takes on all possible values of y over the integral.
<p align="center">
<img src="https://github.com/hanhanwu/Hanhan_Data_Science_Practice/blob/master/Applied_Statistics/Learning_Notes/images/continuous_bayes.PNG" width="300" height="100" />
</p>
  
* Both can be used to calculate probability, and it seems that Bayesian Theoram can be used for wider scenarios.
  * Check some examples [here][7]

## Bayesian Rules Applied in Model and Data
* With Baye's Rule, given the same data, we can check the strength of belief in each model:
  * `P(M1|D)/P(M2|D) = P(D|M1)/P(D|M2) * P(M1)/P(M2)`
  * The ratio of the posterior beliefs is the ratio of evidence * the ratio of priori beliefs.
  * The ratio of evidence is "Bayes Factor", `P(D|M1)/P(D|M2)`.
    * Bayes factor is the equivalent of p-value in the bayesian framework. The null hypothesis in bayesian framework assumes ∞ probability distribution only at a particular value of a parameter (say θ=0.5) and a zero probability else where. The alternative hypothesis is that all values of θ are possible, hence a flat curve representing the distribution.
    * `To reject a null hypothesis, a BF <1/10 is preferred.`
* The posterior is proportional to the product of prior and the likelihood.
  * So the shape of posterior is influenced by both prior and likelihood.
* 3 Major goals of inference
  * Estimate parameter θ
    * Such as P(θ|D, M), given data and model, what the probability of a certain param value
    * `P(θ|D, M) = P(D|θ, M)*P(θ|M)/P(D|M)`
      * `P(D|θ, M)` is the likelihood
      * `P(θ|M)` is the prior
      * `P(D|M)` is the evidence
  * Prediction of data values
  * Model comparison
    * Bayesian method automatically considers the model complexity when assessing to which degree we should believe in the model
    * A complex model has many more available values for θ, so it has higher chance to fit an arbitrary dataset.
* [See my past experience notes here][9]
    
## Bayesian Reasoning in Daily Life
* <b>We have a space of beliefs that are mutually exclusive and exhaust all possibilities.</b> What Bayes’ rule tells us is exactly how much to shift our beliefs across the available possibilities.
### Holmesian Deduction
  * "How often have I said to you that when you have eliminated the impossible, whatever remains, however improbable, must be the truth?"
    * `p(D|θi) = 0 for i != j, then, no matter how small the prior p(θj) > 0 is, the posterior p(θj|D) must equal one.`
  * When the data make some options less believable, we increase belief in the other options.
### Judicial Exoneration
  * It's the reversed Holmesian Deduction.
  * `When p(D|θj) is higher, then even if p(D|θi) is unchanged for all i ! j, p(θi|D) is lower.`
  * When the data make some options more believable, we decrease the belief in the other options.
  
## [Binomial Proportion via Mathematical Analysis][8]

## Bayesian Approach for Hypothesis Analysis
* [Concepts][4]
* [How to use Exploratory (a tool) to do Bayesian A/B test with/without prior][5]
  * [Exploratory][6]

## R Analysis
* [Running Proportion][1]
  * It's flip coin problem, the sequence of binary values. `Running Proportion = cumulated sum / cumulated count`, with running proportion, it can help display the trending of a coin's head (or tail) frequency when the numer of flips is increasing.
  
* [Integral of Density][3]
  * The integral of this normal distribution is the approximation of the sum of width * height from each interval


[1]:https://github.com/hanhanwu/Hanhan_Data_Science_Practice/blob/master/Applied_Statistics/Learning_Notes/bayesian_analysis/bayesian_running_proportion.R
[2]:https://www.amazon.com/Doing-Bayesian-Data-Analysis-Tutorial/dp/0123814855/ref=cm_cr_arp_d_product_top?ie=UTF8
[3]:https://github.com/hanhanwu/Hanhan_Data_Science_Practice/blob/master/Applied_Statistics/Learning_Notes/bayesian_analysis/bayesian_integralOfdensity.R
[4]:https://github.com/hanhanwu/Hanhan_Data_Science_Practice/blob/master/Applied_Statistics/Learning_Notes/bayesian_analysis/Bayesian_Approaches_for%20Hypothesis_Tests.md
[5]:https://blog.exploratory.io/an-introduction-to-bayesian-a-b-testing-in-exploratory-cb5a7ad80963
[6]:https://exploratory.io/?utm_campaign=ab_test&utm_medium=blog&utm_source=medium
[7]:https://brilliant.org/wiki/bayes-theorem/
[8]:https://github.com/hanhanwu/Hanhan_Data_Science_Practice/blob/master/Applied_Statistics/Learning_Notes/bayesian_analysis/binomial_proportion.md
[9];https://github.com/hanhanwu/Hanhan_Data_Science_Resources/blob/master/Experiences.md#naive-bayesian
