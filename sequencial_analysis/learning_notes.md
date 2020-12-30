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
  
  
  
  ## References
  * [Practical Time Series Analysis][1]
  
  
  [1]:https://github.com/PacktPublishing/Practical-Time-Series-Analysis
