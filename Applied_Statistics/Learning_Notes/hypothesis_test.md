# Hypothesis Tests


## Terminology
* Type I Error - When H0 is true but got rejected
* Type II Errle - When H0 is false but got accepted
* `α` - is the probability for type I error
* `β` - is the probability for type II error
* Both `α` and `β` will decrease when sample size increases

  
## Test About 1 Mean
* `α = P(X̄ >= hypothesis_mean_threshold; H0)`
      `= P((X̄ - sample_mean)/(sample_std/sqrt(sample_size)) >= (hypothesis_mean_threshold- sample_mean)/(sample_std/sqrt(sample_size)); H0)`
      `= 1- 𝚽((hypothesis_mean_threshold- sample_mean)/(sample_std/sqrt(sample_size)))`
      `= 1 - 𝚽(𝒛α)`
      `= 1 - The Standard Normal Right-Tail Probabilities of 𝒛α`
* `β = P(X̄ < hypothesis_mean_threshold; H1) = 𝚽(𝒛α)`
* `p-value = P(X̄ >= sample_mean; μ=hypothesis_mean_threshold) `
          `= 1-𝚽((sample_mean - hypothesis_mean_threshold)/(sample_std/sqrt(sample_size)))`
  * If p-value <= α, reject H0. Normally we have `α=0.05, 0.01, 0.1`
  * H0 here is `X̄ >= sample_mean`
* 1 tail vs 2 tails
  * In 1 tail, `H0: X̄ >= sample_mean`; In 2 tails, `H0: X̄ = sample_mean`
  * When it's 2 tails, p-vlaue is the doubled value


