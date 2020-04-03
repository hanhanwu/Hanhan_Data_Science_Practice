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
* About Boxplot
  * IQR (Interquantile Range) = q3 - q1, third quantile - first quantile
  * The rectangle of boxplot has left and right side drawn at q1, q3, with a vertical line segment drawn at median.
    * The length of the box = IQR
    * When the median line is closer to q1 side, the distribution is left skewed; closer to q3 side, the distribution is right skewed.
  * A left whisker is drawn as a horizontal line segment from the minimum to the left side of the box.
  * A right whisker is drawn as a horizontal line segment from the right side of the box to the maximum.
  * 1.5*IQR away from left/right whisker are inner fences, 3*IQR away from left/right whisker are outter fences.
    * Data lies between inner fence and outer fence on left/right side are "suspected outliers", outside of outer fence are "outliers".
    
## Order Statistics
* About q-q plot (quantile-quantile plot)
  * Theory is hard to understand, but the application is easier to understand.
  * The quantile-quantile (q-q) plot is a graphical technique for determining if two data sets come from populations with a common distribution.
  * [Python application of q-q plot][1]
    * With this method we can check whether a dataset has a certain distribution.
  
[1]:https://www.statsmodels.org/dev/generated/statsmodels.graphics.gofplots.qqplot.html
