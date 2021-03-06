# Time Series Notes

## Time Series Plots
* [R Code][1]
  * Seasonal Subseries Plot - is a plot that will show within group patterns between groups patterns
  * In the example of the code, the data has `Year&Month` column, which made yeat and month as continuous numbers, in order to make the plot works better. However I just tried to use `Year` column instead of `Year&Month`, the plot is the same.

## Moving Average
* Take a moving average is a smoothing process
* You can take average as the forecast value ONLY when there is no trend.
  * There are 2 ways to calculate average
    * `mean = sum/count`
    * Moving Average - the average of past successive k values, and keep moving
  * Neither above average methods will forecast well when there is significant trend
    * <b>Double Moving Average</b> can hadle the forecast with trending better
    * This 2 min video is pretty good to learn how does double moving average works: https://www.youtube.com/watch?v=eyQsTQhqSfs
## Exponential Smoothing
* Exponential Smoothing assigns <b>exponentially</b> decreasing weights as the observation get older. That is to say more recent records get higher weights.
  * Single Exponential Smoothing - single coefficient α
    * `St+1 = α*yt + (1-α)*St`
      * St+1 is the forecast result at t+1 time
      * α is coefficient
      * St is the forecast result at t time
      * yt is the real value at t time
    * Choose the better α that can minimize MSE
    * But single conefficient α is not enough to capture the trend
  * Double Exponential Smoothing - coefficient α，γ
    * `St+1 = α*yt + (1-α)*(St + bt)`, 0 <= α <= 1
    * `bt+1 = γ*(St+1 - St) + (1-γ)*bt`, 0 <= γ <= 1
    * The method is trying to capture trend better.
      * The first formula add previous trend value `bt` to the previous forecast value `St`, in order to reduce lag and bring St+1 to proper base of current value
      * The second formula updates the trend based on the last 2 values
    * There are normally 3 methods to calculate `b1`
      * `b1 = y2-y1`
      * `b1 = [(y2-y1) + (y3-y2) + (y4-y3)]/3`
      * `b1 = (yn-y1)/(n-1)`
    * But double exponential smoothing won't work when the time series has seasonality (periodicity)
  * Triple Exponential Smoothing - coefficient α，γ, β
    * It's called "Holt-Winters" (HW) too.
    * `St = α*yt/It-L + (1-α)*(St-1+bt-1)`, overall smoothing
    * `bt = γ(St - St-1) + (1-γ)*bt-1`, trend smoothing
    * `It = β*yt/St + (1-β)*It-L`, seasonal smoothing
    * `Ft+m = (St + m*bt)*It-L+m`, forecast
    * `y` is the observation; `St` is the smoothed observation; `b` is the trend factor; `I` is the seasonal index; `F` is the forecast at `m` periods; A complete season's data has `L` periods
    * It's advisable to use 2 complete seasons (2L periods), in order to estimate trend factor
      * To initial trend factor, `b = ((yL+1 - y1)/L + (yL+2 - y2)/L + ... + (yL+L - yL)/L)/L`
    * To initialize seasonal indices, the example here is good: https://www.itl.nist.gov/div898/handbook/pmc/section4/pmc435.htm
    * When γ, β have at least 1 as 0, doesn't means there is no trend or seasonality, but means the initial values for trend and/or seasonality were correct. No updating was necessary in order to arrive at the lowest possible MSE, but better to inspect the updating formulas to verify this.
## Univariate Time Series Models
* It indicates the time series only has single (scalar) observations recorded sequentially over equal time increments.
* A stationary process has the property that the mean, variance and autocorrelation structure do not change over time. 
* Common Approaches for Univariate Time Series Models
  * Trend, Seasonal and Residual decompositions - such as triple exponential smoothing, HW method
  * Frequency based methods
  * AR model
    * `Xt=δ+ϕ1Xt−1+ϕ2Xt−2+⋯+ϕpXt−1+At`, Xi is time series, At is white noise.
    * AR model is in fact a linear regression of current value of series against one or more previous time series
    * `p` is the order of AR model, `δ=(1−(ϕ1 + ϕ2 + .... + ϕp))*μ`, μ denoting the process mean
  * MA model
    * `Xt=μ+At−θ1At−1−θ2At−2−⋯−θqAt−q`, At-i is white noise, μ is the mean of the series
    * MA model is in fact the linear regression of current value of series against white noise or random shocks of one or more prior values of series
    * `q` is the order of MA model
    * The random shocks at each point are assumed to come from the same distribution, typically a normal distribution, with location at zero and constant scale.
  * Box-Jenkins Method
    * It's a systematic methodology for identifying and estimating models that could incorporate both AR and MA models
    * <b>It assumes the data is stationary</b>, Box and Jenkins also suggests to differencing the non-stationary time series multiple times to achieve stationary
    * Better to have at least 50 ~ 100 observations
    * How to decide model and order (p, q) with autocorrelation plot
      * AR model becomes 0 at lag `p+1` or greater
      * MA model becomes 0 at lag `q+1` or greater
      <img src="https://github.com/hanhanwu/Hanhan_Data_Science_Practice/blob/master/sequencial_analysis/autocorrelation_plots.png" width="400" height="600">
      
    * To validate whether the model is good
      * The residuals should be white noise (or independent when their distributions are normal) drawings from a fixed distribution with a constant mean and variance
      * If you have the ground truth, you can also compare RMSE between the froecast and the prediction
    
## References
* [Engineering Statistics - Time Series][2]
  
[1]:https://github.com/hanhanwu/Hanhan_Data_Science_Practice/blob/master/sequencial_analysis/time_series_plots_R.R
[2]:https://www.itl.nist.gov/div898/handbook/pmc/section4/pmc446.htm
