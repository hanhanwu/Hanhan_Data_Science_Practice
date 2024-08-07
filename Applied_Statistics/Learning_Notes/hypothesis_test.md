# Hypothesis Tests

## Examples
* [Example - Calculate type I, II error][1]

## Terminology
* Type I Error - When H0 is true but got rejected
* Type II Errle - When H0 is false but got accepted
* `α` - is the probability for type I error
* `β` - is the probability for type II error, also called as "power"
* Both `α` and `β` will decrease when sample size increases
* `𝒛α` is the z-value of `α`
  * 95% confidence, 1-α = 0.95, so α/2 = 0.025, 𝒛0.025 = 1.96
  * 90% confidence, 1-α = 0.90, so α/2 = 0.05, 𝒛0.05 = 1.645

  
## Test About 1 Mean
### Method 1 - Using p-value
* <b>Smaller p-value is, the less we believe in H0</b>
* `α = P(X̄ >= μ0; H0)`
      `= P((X̄ - μ)/(s/sqrt(n)) >= (μ0- μ)/(s/sqrt(n)); H0)`
      `= 1- 𝚽((μ0 - μ)/(s/sqrt(n)))`
      `= 1 - 𝚽(𝒛α)`
      `= 1 - The Standard Normal Right-Tail Probabilities of 𝒛α`
  * `μ` is sample mean
  * `μ0` is hypothesis mean threshold
  * `s` is sample standard deviation
  * `n` is sample size
* `β = P(X̄ < μ0; H1) = 𝚽(𝒛α)`
* `p-value = P(X̄ >= μ; μ=μ0) `
          `= 1-𝚽((μ - μ0)/(s/sqrt(n)))`
  * If p-value <= α, reject H0. Normally we have `α=0.05, 0.01, 0.1`
  * H0 here is `X̄ >= μ`
  
### 1 tail vs 2 tails
* When it's 2 tails, p-vlaue is the doubled value

### Method 2 - Critical Region
* <b>If the condition of critical region has been satisfied, we need to accept H1 (or not accept H0).</b> Sometimes also needs to consider approximation, without rejecting H0 strictly.
* Either we can calculate t-value to decide whether to accept H0. However sometimes, using t-value, it might accept H0 at higher confidence level (such as 99%) but reject at lower confidence level (such as 95%).
* Or we calculate the confidence interval (the non-critical region for μ)
* Variance Known
<p align="left">
<img width="400" height="200" src="https://github.com/hanhanwu/Hanhan_Data_Science_Practice/blob/master/Applied_Statistics/Learning_Notes/images/t_test_1mean_known_var.png">
  
* Variance Unknown
<p align="left">
<img width="400" height="200" src="https://github.com/hanhanwu/Hanhan_Data_Science_Practice/blob/master/Applied_Statistics/Learning_Notes/images/t_test_1mean_unknown_var.png">
</p>

* Independent vs Dependent
  * If X and Y are independent, we have above `μ0`
  * If X and Y are dependent, `μ0=0`, so it will be comparing `μ` with 0
   * An dependent example can be, comparing the time "before" vs "after"


## Test About Proportions
* One Proportion
<p align="left">
<img width="400" height="200" src="https://github.com/hanhanwu/Hanhan_Data_Science_Practice/blob/master/Applied_Statistics/Learning_Notes/images/test_proportions1.png">
 </p>
 
 * Two proportions
   * Y1 and Y2 represent, respectively, the numbers of observed successes in n1 and n2 independent trials with probabilities of success p1 and p2. 
   * `p̂1 = Y1/n1`, read as "p1 hat"
   * `p̂2 = Y2/n2`
   * `p̂ = (Y1+Y2)/(n1+n2)`
 <p align="left">
<img width="400" height="200" src="https://github.com/hanhanwu/Hanhan_Data_Science_Practice/blob/master/Applied_Statistics/Learning_Notes/images/test_proportions2.png">
 </p>
 
 
## Wilcoxon Test
* It's test about mean.
* When normality assumption no longer met, it needs nonparametric method (distribution free).
* Wilcoxon statistic W is the sum of these n signed ranks, and therefore is often called the Wilcoxon signed rank statistic.
Here's an example about how to calculate Wilcoxon Test:
 <p align="left">
<img width="400" height="200" src="https://github.com/hanhanwu/Hanhan_Data_Science_Practice/blob/master/Applied_Statistics/Learning_Notes/images/wilcoxon_statistics.png">
 </p>
 
* `z = (W-0)/sqrt(n*(n+1)*(2n+1)/6)` is approximately N(0,1)
  * When it's N(m, std), it should be `W-m`
  * When `z >= zα/2` or `W>= zα/2*sqrt(n*(n+1)*(2n+1)/6)` then reject H0
    * So after calculating z, you calculate W and compare


## Best Critical Region
<p align="left">
<img width="600" height="200" src="https://github.com/hanhanwu/Hanhan_Data_Science_Practice/blob/master/Applied_Statistics/Learning_Notes/images/best_critical_region.png">
 </p>
 
 <p align="left">
<img width="600" height="300" src="https://github.com/hanhanwu/Hanhan_Data_Science_Practice/blob/master/Applied_Statistics/Learning_Notes/images/npl.png">
 </p>
 
 * A test defined by a critical region C of size α is a uniformly most powerful test if it is a most powerful test against each simple alternative in H1. The critical region C is called a uniformly most powerful critical region of size α.


## Likelihood Ratio Test
* We assume that the functional form of the pdf is known, the pdf of X is f(x;θ), where `θ` represents one or more unknown parameters.
* Let `𝛀` denote the total parameter space, `𝛚` is a subset of 𝛀 and `𝛚'` is the complement of 𝛚 with respect to 𝛀

<p align="left">
<img width="700" height="200" src="https://github.com/hanhanwu/Hanhan_Data_Science_Practice/blob/master/Applied_Statistics/Learning_Notes/images/likelihood_ratio.png">
 </p>
 
 <p align="left">
<img width="700" height="200" src="https://github.com/hanhanwu/Hanhan_Data_Science_Practice/blob/master/Applied_Statistics/Learning_Notes/images/likelihood_ratio_test.png">
 </p>


[1]:https://www.youtube.com/watch?v=BJZpx7Mdde4
