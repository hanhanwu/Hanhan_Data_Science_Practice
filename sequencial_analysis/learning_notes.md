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
      * It may not have a fixed period of variations
      * The average periodicity for cyclical changes can be larger (such as in years), while seasonal variations can happen within the same year
      * This is not the part to be removed when making the ts stationary, since it's not time dependent and can only be explained by exogenous variables, when we check stationary, exogenous variables are not considered.
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
  * Stationary ts is characterized by, constant internal structures (mean, variance, autocorrelation) that do not change over time (time independent)
  * Autocorrelation reflects the degree of linear dependency between ith time series and i+h or i-h time series (h is the lag)
    * Because it's time independent, it ca be reliably used to forecast future time series
    * Positive autocorrelation means the 2 time series moves towards the same direction; negative means opposite directions; 0 means the temporal dependencies is hard to find
    * Autocorrelation is a value between [-1, 1]
  * ACF plot
    * Each vertical bar indicates the autocorrelation between ith ts and i+gth ts, given a confidence level (such as 95%), out of confidence interval (the threshold lines), the autocorrelation is significant
  * PACF
    * In ACF, the autocorrelation between ith ts and i+gth ts can be affected by i+1th, t+2th, ..., i+g-1th ts too, so PACF removes the influences from these intermediate variables and only checks the autocorrelation between ith ts and i+gth ts
    * Lag0 always has autocorrelation as 1
    * In the example [here][2], besides lag0, only at lag1 there is significant autocorrelation, so the order for AR model is 1
  
* Moving Statistics
  * window size w: t1, t2, ..., tw
  * stride length l: (t1, t2, ..., tw) a window and its next window will (t1+l, t2+l, ..., tw+l)
    * When using python `rolling()` function, by default the stride length is 1, to get n (n>1) stride length, just removed first n records from `rolling()` results
  * Moving averages have an effect of smoothing the original time series by eliminating random noise
  * [Example of nxm weighted moving average][5]
    * rolling by m, then rolling by n again
    * This method is a way to make ts stationary
    
* Exponential Smoothing Methods
  * The limitation of moving average and weighted moving average is the ignorance of observation recency effect, and exponential smoothing methods can help deal with this issuer by having exponential decaying weights on observations (older data gets lower weights)
  * <b>Need to convert the ts to stationary</b> before applying moving average and exponential smoothing methods, since that align with the assumption of these methods
  * Smoothing methods are used to remove random noise, but can be extended for forecasting by ading smoothing factor α, trend factor β, seasonality factor γ in exponential smoothing methods
  * First order exponential smoothing
    * `F_t = α*x_t + (1-α)*F_t-1`, x_i is observation value, α is the smoothing factor in [0,1] range
      * When α=0, the forecasted ts is a constant line, 0 variance
      * When α=1, the forecasted ts is the right shift of the original ts by 1 lag, same variance as the actual ts
      * When α is increasing from 0 to 1, the variance of the forecasted ts is also exponentialy increasing, from 0 to the actial ts' variance
    * [Python implementation of this method][9]
    * It's also called as Holt-Winters foreacsting, check how did I use the built-in function [here][10]
  * Second order exponential smoothing
    * `F_t = α*x_t + (1-α)*(F_t-1 + T_t-1)`
    * `T_t = β*(F_t - F_t-1) + (1-β)*T_t-1`
      * β is the trend factor in [0,1] range
      * Second order can capture the variation of the real signal better than first order if the trend component is not constant
    * [Python implementation of this method][11]
  * Triple order exponential smoothing
    * `F_t = α*(x_t - S_t-L) + (1-α)*(F_t-1 + T_t-1)`
    * `T_t = β*(F_t - F_t-1) + (1-β)*T_t-1`
    * `S_t = γ(x_t - F_t) + (1-γ)S_t-C`
      * γ is the seasonality factor in [0,1] range
    * [Python implementation of this method][12]

    
* Methods to Convert to Stationary ts
  * I often use both Augumented Dickey-Fuller (ADF) and KPSS to check stationary, [see this example][3]
    * ADF checks differencing stationary, which based on the idea of differencing to transform the original ts
      * `autolag='AIC'` instructs the function to choose a suitable number of lags for the test by maximizing AIC
    * KPSS checks trending stationary
    * And I prefer to check Test Statistic and critical values instead of checking p-value only, since in this way we can get the confidence level of stationary
  * Differencing methods to convert to stationary
    * First-order differencing: take differences between successive realizations of the time series
      * `x_t - x_t-1`
    * Second-order differencing
      * `(x_t - x_t-1) - (x_t-1 - x_t-2)`
    * Seasonal differencing
      * `x_t - x_t-m`
      * If in the de-trended ts' ACF plot, we are seeing repeated significant autocorrelation (beyond the confidence interval)
      * [An example of seasonal differencing][4], just use `diff()`
    * Weighted moving averages
      * The nxm weighted moving averages method above can help transform ts into stationary
  * Decomposition methods to convert to stationary
    * [Example of decomposing a ts][7]
      * Different from the above decomposition which can be used on the original ts, [Prophet's decomposition comes with the frecasting model][8]
    * Additive model
      * x_t = F_t + S_t + E_t
      * This model is usually applied when thre is a time-dependent trend cycle component but independent seasonality that does not change over time
    * Multiplicative model
      * x_t = F_t * S_t * E_t
      * This model often used when there is time-varying seasonality
    * [Example of applying both additive and multiplicative methods for decomposition, and python built-in `seasonal_decompose`][6]
  
  
## References
* [Practical Time Series Analysis][1]

  
  
[1]:https://github.com/PacktPublishing/Practical-Time-Series-Analysis
[2]:https://github.com/PacktPublishing/Practical-Time-Series-Analysis/blob/master/Chapter01/Chapter_1_Autocorrelation.ipynb
[3]:https://github.com/hanhanwu/Hanhan_Break_the_Limits/blob/master/Bank_Fantasy/Golden_Bridge/stationary_analysis.ipynb
[4]:https://github.com/PacktPublishing/Practical-Time-Series-Analysis/blob/master/Chapter02/Chapter_2_Seasonal_Differencing.ipynb
[5]:https://github.com/PacktPublishing/Practical-Time-Series-Analysis/blob/master/Chapter02/Chapter_2_Moving_Averages.ipynb
[6]:https://github.com/PacktPublishing/Practical-Time-Series-Analysis/blob/master/Chapter02/Chapter_2_Time_Series_Decomposition_using_statsmodels.ipynb
[7]:https://github.com/hanhanwu/Hanhan_Break_the_Limits/blob/master/Bank_Fantasy/Golden_Bridge/seasonal_decomposing.ipynb
[8]:https://github.com/hanhanwu/Hanhan_Break_the_Limits/blob/master/Bank_Fantasy/Golden_Bridge/prophet_forecast.ipynb
[9]:https://github.com/PacktPublishing/Practical-Time-Series-Analysis/blob/master/Chapter03/Chapter_3_simpleExponentialSmoothing.py
[10]:https://github.com/hanhanwu/Hanhan_Data_Science_Practice/blob/master/sequencial_analysis/time_series_forecasting.ipynb
[11]:https://github.com/PacktPublishing/Practical-Time-Series-Analysis/blob/master/Chapter03/Chapter_3_doubleExponentialSmoothing.py
[12]:https://github.com/PacktPublishing/Practical-Time-Series-Analysis/blob/master/Chapter03/Chapter_3_tripleExponentialSmoothing.py
