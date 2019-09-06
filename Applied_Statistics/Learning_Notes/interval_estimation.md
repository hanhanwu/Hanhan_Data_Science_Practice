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
