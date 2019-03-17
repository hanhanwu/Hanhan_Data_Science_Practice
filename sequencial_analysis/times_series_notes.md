# Time Series Notes

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
    
  
