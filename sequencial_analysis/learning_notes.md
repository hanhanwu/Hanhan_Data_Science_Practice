# Learning Notes 

## Concepts

* Components if a time series
  * f: trend
    * For example, there is linear trending, we will see ts goes up or down, using a linear regression model can get the trend line with coefficient and intercept 
  * s: seasonality
    * It manifests as repetitive and period variations in a ts
    * For example, a seasonality series can be a function of sine wave (`sin()`) with residuals 
    * "seasonal sub series" plots the mean or variance of de-trended residuals (seasonality + residuals) in each season (such as Q1, Q2, etc.)
  * c: cyclical
    * Comparing with seasonality:
      * It occurs less frequently than seasonal fluctuations
      * It may not have a fixed period of avriations
      * The average periodicity for cyclical changes can be larger (such as in years), while seasonal variations can happen within the same year
  * e: residuals
    * Irreducible error component, random and doesn't systematic dependent on the time index. It's caused by the lack of info of explanatory variables that can model the variations, or caused by random noise
    
* Time Series Models
  * Zero Mean models: The model has constant mean and constant variance
    * Observations are assumed to be `iid` (independent and identically distributed), and represent the random noise around the same mean
    * `P(X1=x1, X2=x2, ..., Xn=xn) = f(X1=x1)*f(X2=x2)*...*f(Xn=xn)`
  * Random Walk models: the cumulative sum of the zero mean model (a sum of n_i iids at ith iid), and it has 0 mean and constant variace
    * So if we take the difference between 2 consecutive time indices from this model, we will an iid with 0 mean and constant variance, which is also a zero mean model
  * Trend models: `x_t = μ_t + y_t`
    * `μ_t` is the time dependent trend of the series, it can be linear or non-linear
    * `y_t` is zero mean model, residuals
  * Seasonality models: `x_t = s_t + y_t`
    * `s_t` is a sum of weighted sum of sine waves (both `sin` and `cos`)
    
* ACF and PACF (Autocorrelation and Partial Autocorrealtion)
  * Stationary ts is characterized by, constant mean and correlation that depends only on the time lag between 2 time steps, but independent of the value of the time step.
  * Autocorrelation reflects the degree of linear dependency between ith time series and i+h or i-h time series (h is the lag)
    * Because it's time independent, it ca be reliably used to forecast future time series
    * Positive autocorrelation means the 2 time series moves towards the same direction; negative means opposite directions; 0 means the temporal dependencies is hard to find
    * Autocorrelation is a value between [-1, 1]
  * ACF plot
    * Each vertical bar indicates the autocorrelation between ith ts and i+gth ts, given a confidence level (such as 95%), out of the threshold lines, the autocorrelation is significant
  * PACF
    * In ACF, the autocorrelation between ith ts and i+gth ts can be affected by i+1th, t+2th, ..., i+g-1th ts too, so PACF removes the influences from these intermediate variables and only checks the autocorrelation between ith ts and i+gth ts
    * Lag0 always has autocorrelation as 1
    * In the example [here][2], besides lag0, only at lag1 there is significant autocorrelation, so the order for AR model is 1
  
  
  
  ## References
  * [Practical Time Series Analysis][1]

  
  
  [1]:https://github.com/PacktPublishing/Practical-Time-Series-Analysis
  [2]:https://github.com/PacktPublishing/Practical-Time-Series-Analysis/blob/master/Chapter01/Chapter_1_Autocorrelation.ipynb
