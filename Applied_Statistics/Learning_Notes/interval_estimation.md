# Interval Estimation

## Terminology/Symbols
* `α` - Often used as a `100*(1-α)%` confience interval. Such as "a 95% confidence interval". `1-α` is called as <b>confience conefficient</b>
* `x̅` - sample mean
* `N(μ， σ2)` - normal distribution has mean as `μ`， standard deviation as `δ`
* `n` - random sample size
* `𝒛α/2` - z value for α/2
  * 95% confidence, 1-α = 0.95, so α/2 = 0.025, `𝒛α/2` = 1.96
  * 90% confidence, 1-α = 0.90, so α/2 = 0.05, `𝒛α/2` = 1.645
* `tα/2(n-1)` is the t value for α/2 with n-1 degree of freedom
* `δw = sqrt(power(δx, 2)/n + power(δy, 2)/m)`, `δw` is the satndard deviation of `x̅ - ȳ`

## Confidence Intervals for Means

🌺 <b>The intervals calculated here represents - the intervals that include the unknown mean `μ`.</b>
* For example, with the confidence interval, you can say there is 100(1-α)% confidence that the population mean is within this interval

### Type 1 Interval
* `[x̅ - 𝒛α/2 * δ/√n, x̅ + 𝒛α/2 * δ/√n]` is a confidence interval 100(1-α)% for μ
* Often used when
  * sample mean `x̅` is known, standard deviation `δ` is known, and there is enough sample size (n > 30)
  
### Type 2 Interval
* `[x̅ - 𝒛α/2 * S/√n, x̅ + 𝒛α/2 * S/√n]` is a confidence interval 100(1-α)% for μ
* Often used when
  * sample mean `x̅` is known, standard deviation `δ` is unknown, but there is enough sample size (n > 30), the distribution is approximate to normal distribution, `S` is the standard deviation of the sample
  
### Type 3 Interval
* `[x̅ - tα/2(n-1) * S/√n, x̅ + tα/2(n-1) * S/√n]` is a confidence interval 100(1-α)% for μ
* Often used when
  * sample size is small (n<30), standard deviation `δ` is unknown

## Confidence Intervals for the Difference of 2 Means

🌺 <b>The intervals here represents - the intervals of `x̅ - ȳ` the difference of the means of 2 normal distribution.</b>

### Type 1 Interval
* `[x̅ - ȳ - 𝒛α/2 * δw, x̅ - ȳ + 𝒛α/2 * δw]` is a confidence interval 100(1-α)% for `μx - μy`
* Often used when
  * sample mean `x̅`, `ȳ` are known, standard deviation `δw` is known, and there is enough sample size (n > 30)
  
### Type 2 Interval
* `[x̅ - ȳ - 𝒛α/2 * δs, x̅ - ȳ + 𝒛α/2 * δs]` is a confidence interval 100(1-α)% for `μx - μy`
  * `δs = sqrt(power(sx, 2)/n + power(sy, 2)/m)`
* Often used when
  * sample mean `x̅`, `ȳ` are unknown, sample size is large enough, therefore, we can replace `δx`, `δy` with sample standard deviation `sx` and `sy`
  
### Type 3 Interval
* `[x̅ - ȳ - t0 * Sp * sqrt(1/n + 1/m), x̅ - ȳ + t0 * Sp * sqrt(1/n + 1/m)]`
  * `t0 = tα/2 * (n+m-2)`, t distribution
  * `Sp = sqrt(((n-1) * power(Sx, 2) + (m-1) * power(Sy, 2))/(n+m-2))`
* Often used when
  * Sample sizes are small and `δx`, `δy` are unknown but equal
  
### Type 4 Interval
* `[x̅ - ȳ - t0 * Sp * sqrt(1/n + 1/m), x̅ - ȳ + t0 * Sp * sqrt(1/n + 1/m)]`
  * `t0 = tα/2 * r`, t distribution, r is no longer `n+m-2`
    * `r = power((power(sx, 2)/n + power(sy, 2)/m), 2) / (power(power(sx, 2)/n, 2)/(n-1) + power(power(sy, 2)/m, 2)/(m-1))`
    * This is because, the smaller sample size is associated with the larger variance by greatly reducing the number of degrees of freedom from the usual `n + m − 2`. 
  * `Sp = sqrt(((n-1) * power(Sx, 2) + (m-1) * power(Sy, 2))/(n+m-2))`
* Often used when
  * Sample sizes are small and `δx`, `δy` are unknown but unequal
  
### Type 5 Interval
* `[D̄-tα/2(n-1)*sd/√n, D̄+tα/2(n-1)*sd/√n]`
* Often used when
  * X, Y came from the same data sample, but repersents "before", "after" results


## Confidence Itervals for Proportions
🌺 <b>It calculates the confidence of intervals of proportion p.</b> `p` here represents how frequent event Y will happen.

### Type 1 Interval
* `[y/n - 𝒛α/2 * sqrt((y/n)*(1-y/n)/n), y/n + 𝒛α/2 * sqrt((y/n)*(1-y/n)/n)]`
  * `y` means the number of records when event Y happened in sample `n`
  * `y/n` is estimated `p`
  * The interval formula means, we are 100(1-α)% confident that p is within `𝒛α/2 * sqrt((y/n)*(1-y/n)/n)` of estimated `p`
* Often used when
  * Sample size `n` is large
* One Side Confidence Interval
  * `[0, y/n + 𝒛α/2 * sqrt((y/n)*(1-y/n)/n)]` is the upper bound for p
  * `[y/n - 𝒛α/2 * sqrt((y/n)*(1-y/n)/n), 1]` is the lower bound for p
  
### Type 2 Interval
* `[p' - 𝒛α/2 * sqrt(p'*(1-p')/(n+4))]`
  * `p' = (y+2)/(n+4)`, it's the biased estimator of `p`, but it is the Bayes shrinkage estimator if we use the beta prior pdf with parameters `α=2, β=2`
* Often used when
  * Sample size `n` is small, and Y or n-Y many not happen within the sample, therefore with type 1 interval above, the calculated result will be 0
  
### Confidence Interval for Proportion Difference
* `[y1/n1 - y2/n2 - 𝒛α/2 * sqrt((y1/n1)*(1-y1/n1)/n1 + (y2/n2)*(1-y2/n2)/n2)]`
  * It's the interval for `p1-p2`

## Sample Size
🌺 <b>The problem its trying to solve is, given the maximum error of estimate and confidence conefficient, to estimate the sample size needed in order to estimate a mean.</b>

* Smaller the variance, smaller the sample size is needed.
  * An extreme example, when `δ=0`, you just need 1 record as the sample
* An estimate associated with longer confidence interval with a smaller conffidence coefficient is satisfactory, and therefore a smaller sample size is needed.
* `n = power(𝒛α/2, 2) * power(δ, 2) / power(ε, 2)`
  * `ε = 𝒛α/2 * δ / sqrt(n)` is the maximum error of estimate
* `n = power(𝒛α/2, 2)/(4 * power(ε, 2))`
  *  If we want the 100(1 − α)% confidence interval for p to be within `[y/n - ε, y/n + ε]`
  * Often, we don't have a strong prior idea about `p`, it's within `[0, 1]`, assume `p'` is close to `p`, and there is always `p' * (1-p') <= 1/4`. `n = power(𝒛α/2, 2) * p' * (1-p') / power(ε, 2)`
  * Used when n can be large (total population N is large)
* `n = m/(1 + (m-1)/N)`
  * `m = power(𝒛α/2, 2) * p' * (1-p') / power(ε, 2)`, whe we don't know `p'`, give it value as 1/2
  * Used when the total population N is limited

## Distribution Free Confidence Intervals
🌺 <b>It uses order statistics to construct confidence intervals for unknown distribution percentiles.</b>

### Benefits of Distribution Free
* With distribution free method, we only need to assume the distribution is of the continuous type.
* If the distribution is highly skewed or heavy-tailed so that outliers could exist, a distribution-free technique is safer and much more robust.
* With this method, we can also get confidence intervals for various percentiles.

### Order Statistics
* It just means to put a list of numbers in order.
* For example
  * We have a list of values [3,1,5,6,2]
  * Its order statistics is 1<2<3<5<6

### Unknown Distribution Median
* `P(Yi < m < Yj) = (n!/(n-i)!)*power(0.5, i)* power(0.5, n-i) + ... + (n!/(n-j+1)!)*power(0.5, j-1)* power(0.5, n-j+1) = 1- α`
  * `1 − α`, which is the sum of probabilities from a binomial distribution, can be calculated directly or approximated by an area under the normal pdf, provided that n is large enough.
  * The observed interval (yi, yj) could then serve as a 100(1 − α)% confidence interval for the <b>unknown distribution median</b> `m`.
* If the sample arises from a distribution that does not deviate too much from the normal, the confidence interval based upon `x̅` is much shorter.

### Unknown Distribution pth Percentile
* If there are `n` records, then pth percentile is `(n+1)*p`th value
  * For example
    * There are 27 values, 25th percentile is `(27+1)/4 = 7` 7th value is what we want to estimate
    * `1-α = P(Y4 < π0.25 < Y10)`
      * Here, from 7, move left we got 4, move right we got 10. It doesn't have to be these 2 values, it can be (3, 11) and so on. Depends on how much confidence you want to achieve.

## Resampling Method
* In fact, it emphasised on bootstrap. By doing resampling with replacement, it can help us estimate the confidence interval for population mean `μ`, especially in situations when we don't know the underlying distribution.
  * "Resampling methods, however, are used for more than showing that statistics have certain distributions. Rather, they are needed in finding approximate distribu- tions of certain statistics that are used to make statistical inferences."
  * Sample mean has an approximate normal distribution, after going through multiple rounds of resampling.
* To make it easier to understand, let's use an example:
  * You have a sample of 16 values, with sample mean `x̅` as 10.3
  * Without knowing the underlying distribution, we use the emperical distribution as the best guess, so that we have:
    * Distribution T as `T = (X̄ - 10.3)/(S/sqrt(16))`
  * Then we use bootstrap, to choose the samples with 16 values over and over again, for example, we have done resampling 1000 times
  * Plot the 1000 observations of T in histogram, and q-q plot for N(0,1) quantiles vs T quantiles
  * If the confidence level is 95%, we check the t-value at 2.5% percentile, 97.5% percentile of the 1000 values are `c`, `d`
  * Then we have `P(c <= (X̄ - μ)/(S/sqrt(16)) <= d)≈ 0.95`
  * `[x̅ - d*s/sqrt(16), x̅ - c*s/sqrt(16)]` leads to the approximate 95% confidence interval
    * `x̅` is the sample mean of the original 16 values, `s` is the sample standard deviation for them

## Other Notes About Confidence Interval
### Confidence Interval vs Point Estimates
* Point Estimates - It generates an estimated value of population parameter
  * Methods of moments, maximum likelihood estimator are the methods of  point estimates
* Confidence Interval - It generates an estimated value range of population parameter
* Confidence interval is often prefered, since it tells how likely the value range contains the population parameter
  * `α`: is called as "the level of significance"
  * `1-α`: is called as "confidence level" or "confidence coefficient", this is the likelihood or probability
