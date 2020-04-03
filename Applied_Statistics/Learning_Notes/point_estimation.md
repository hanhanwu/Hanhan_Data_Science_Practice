# Point Estimation

## Maximum Likelihood Estimation
🌺 The purpose is to find the most likely estimated PDF parameter `θ`, this estimated value is `θ_hat`.

* Likelihood function `L(θ) = f(x1, θ)*f(x2, θ)*f(x3, θ)*....*f(xn, θ)`
  * `f(xi, θ)` is the PDF (probability distribution function) of the observation xi
* To get maximum likelihood estimator for `θ`:
  * Step 1 - Take natural log of L(θ), ln(L(θ)).
    * Because natural log is strictly increasing, it's easier to find the estimated parameter `θ_hat` that can maximize ln(L(θ)), the results will be the same as maximize L(θ).
  * Step 2 - Take derivate of ln(L(θ))
    * `ln'(L(θ)) = 0`, calculated value is `θ_hat`, the estimated `θ`.
* We repeat the experiment n times, observing sample X1, X2, ... Xn. `u(X1, X2, ... Xn)`is the function used to estimate `θ`. When `θ_hat`=`θ`, we say `u(X1, X2, ... Xn) is the unbiased estimator, otherwise it's biased.

## Exploratory Data Analysis
