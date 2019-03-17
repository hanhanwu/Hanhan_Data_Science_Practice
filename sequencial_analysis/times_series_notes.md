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
  
