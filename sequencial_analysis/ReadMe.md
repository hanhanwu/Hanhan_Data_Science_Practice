I'm planning to practice more on time series analysis, pattern matching and sequential pattern matching.


******************************************************************************************

-- TIME SERIES

* For previous summarized knowledge, check [-- Time Series section][1]
* [AR, MA example in ACF, PACF][2], this in fact is a good example. Use ACF to find MA model, use PACF to find AR model
  * [reference][3]
    * It also includes <b>parameter estimation</b>, `fit <- arima(data,order=c(p,d,q))`
    * <b>Diagnostic Checking</b>, to check non-randomness with residuals, `tsdiag(fit)`. The residuals of a “correctly specified” model should be independently distributed, otherwise it's the wrong model
    * <b>Predict Future Values of Time Series</b>, `LH.pred<-predict(fit,n.ahead=8)`



[1]:https://github.com/hanhanwu/Hanhan_Data_Science_Resources/blob/master/README.md
[2]:https://github.com/hanhanwu/Hanhan_Data_Science_Practice/blob/master/sequencial_analysis/ARMA_example.png
[3]:https://github.com/hanhanwu/Hanhan_Data_Science_Practice/blob/master/sequencial_analysis/ts_r_intro.pdf
