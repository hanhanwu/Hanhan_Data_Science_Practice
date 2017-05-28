I'm planning to practice more on time series analysis, pattern matching and sequential pattern matching.


******************************************************************************************

-- TIME SERIES

* For previous summarized knowledge, check [-- Time Series section][1]
* [AR, MA example in ACF, PACF][2], this in fact is a good example. Use ACF to find MA model, use PACF to find AR model
  * [reference][3]
    * It also includes <b>parameter estimation</b>, `fit <- arima(data,order=c(p,d,q))`
    * <b>Diagnostic Checking</b>, to check non-randomness with residuals, `tsdiag(fit)`. The residuals of a “correctly specified” model should be independently distributed, otherwise it's the wrong model
    * <b>Predict Future Values of Time Series</b>, `LH.pred<-predict(fit,n.ahead=8)`
 
 
-- PRACTICE

* [Previous practice][6]
  * [reference][7]

* [time series beginner][4]
  * [Watch the free videos here FIRST!][8]
    * Often time series are generated as `Xt=(1+pt)Xt−1`, meaning the value of the time series observed at time t equals to the value observed at time t-1, and a small percent change pt at time t. pt is often refered to as <b>return/growth rate</b> of time series, and <b>it should be stable</b>. There is also `Yt=logXt−logXt−1≈pt`. In R, pt is calculated as `plot(diff(log(x)))`
    * <b>Stationary</b>: The <b>mean</b> is constant over time; The <b>correlation structure</b> remains constant over time
    * To convert non-stationary to stationary: Simple differencing can help remove the trend (detrend); You can also use logging + differencing, first logging will tablize the variance, then differencing will do detrend. In R, differencing is using `diff()`, logging is using `log()`. <b>To sum up, logging against Heteroscedasticity, differencing against the trend of the mean</b>.
  * [Reference][5]
  * This reference is really messy, it only chose good examples without dealing with a complete real world problem. I disagree with its method that trying to fit regression for `log(jj)`, because `log(jj)` does not satisfy the stationary for time series. Then off course it will show not a good fit...
  * What I have learned:
    * Different from [my previous practice][6], which was using `ariam` to fit, I'm learning `sariam` for both fit and forecast


[1]:https://github.com/hanhanwu/Hanhan_Data_Science_Resources/blob/master/README.md
[2]:https://github.com/hanhanwu/Hanhan_Data_Science_Practice/blob/master/sequencial_analysis/ARMA_example.png
[3]:https://github.com/hanhanwu/Hanhan_Data_Science_Practice/blob/master/sequencial_analysis/ts_r_intro.pdf
[4]:https://github.com/hanhanwu/Hanhan_Data_Science_Practice/blob/master/sequencial_analysis/time_series_beginner.R
[5]:http://www.stat.pitt.edu/stoffer/tsa4/R_toot.htm
[6]:https://github.com/hanhanwu/Hanhan_Data_Science_Practice/blob/master/time_series_predition.R
[7]:https://www.analyticsvidhya.com/blog/2015/12/complete-tutorial-time-series-modeling/?utm_content=buffer529c5&utm_medium=social&utm_source=facebook.com&utm_campaign=buffer
[8]:https://www.datacamp.com/courses/arima-modeling-with-r
