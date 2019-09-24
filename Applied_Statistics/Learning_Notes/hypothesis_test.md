# Hypothesis Tests


## Terminology
* Type I Error - When H0 is true but got rejected
* Type II Errle - When H0 is false but got accepted
* `α` - is the probability for type I error
* `β` - is the probability for type II error
* Both `α` and `β` will decrease when sample size increases

  
## Test About 1 Mean
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
* 1 tail vs 2 tails
  * When it's 2 tails, p-vlaue is the doubled value
* H1 for `H0: μ = μ0`
  * `H1: μ > μ0`
    * `𝒛 >= 𝒛α` at a significance level α
    * `x̅ >= μ0 + zα(σ/√n)`
  * `H1: μ < μ0`
    * `𝒛 <= 𝒛α`
    * `x̅ <= μ0 - zα(σ/√n)`
  * `H1: μ != μ0`
    * `|𝒛| >= 𝒛α/2`
    * `|x − μ0| >= zα/2(σ/√n )`


