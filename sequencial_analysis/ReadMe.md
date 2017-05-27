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
