# Hypothesis Tests


## Terminology
* Type I Error - When H0 is true but got rejected
* Type II Errle - When H0 is false but got accepted
* `α` - is the probability for type I error
* `β` - is the probability for type II error
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
