# Interval Estimation

## Confidence Intervals for Means

The intervals calculated here means - the intervals that include the unknown mean `μ`.

### Terminology/Symbols
* `α` - Often used as a `100*(1-α)%` confience interval. Such as "a 95% confidence interval". `1-α` is called as <b>confience conefficient</b>
* `x̅` - sample mean
* `N(μ， σ2)` - normal distribution has mean as `μ`， standard deviation as `δ`
* `n` - random sample size
* `𝒛α/2` - z value for α/2
  * 95% confidence, 1-α = 0.95, so α/2 = 0.025, `𝒛α/2` = 1.96
  * 90% confidence, 1-α = 0.90, so α/2 = 0.05, `𝒛α/2` = 1.645
* `tα/2(n-1)` is the t value for α/2 with n-1 degree of freedom

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
