I'm planning to practice more on time series analysis, pattern matching and sequential pattern matching.


******************************************************************************************

TIME SERIES CONCEPTS

* For previous summarized knowledge, check [-- Time Series section][1]
* ARIMA is an acronym that stands for <b>AutoRegressive Integrated Moving Average</b>. It is a class of model that captures a suite of <b>different standard temporal structures</b> in time series data.
* AR: Autoregression. A model that uses the dependent relationship between an observation and some number of lagged observations.
* I: Integrated. The use of differencing of raw observations (e.g. subtracting an observation from an observation at the previous time step) in order to make the time series stationary.
* MA: Moving Average. A model that uses the dependency between an observation and a residual error from a moving average model applied to lagged observations.
* <b>Params (p,d,q)</b> are used to determine which specific ARIMA model will be used
  * <b>p</b>: The number of lag observations included in the model, also called the lag order.
  * <b>d</b>: The number of times that the raw observations are differenced, also called the degree of differencing.
  * <b>q</b>: The size of the moving average window, also called the order of moving average.
  * When set 0 for a param here, means not using this param. Ans therefore, we can simply use AR, I or MA model
* A linear regression model is constructed including the specified number and type of terms, and the data is prepared by a degree of differencing in order to make it stationary, i.e. to remove trend and seasonal structures that negatively affect the regression model.
* Adopting an ARIMA model for a time series assumes that the underlying process that generated the observations is an ARIMA process.
* [VISUALIZED AR, MA example in ACF, PACF][2], this in fact is a good example. Use ACF to find MA model, use PACF to find AR model
  * [reference][3]
    * It also includes <b>parameter estimation</b>, `fit <- arima(data,order=c(p,d,q))`
    * <b>Diagnostic Checking</b>, to check non-randomness with residuals, `tsdiag(fit)`. The residuals of a “correctly specified” model should be independently distributed, otherwise it's the wrong model
    * <b>Predict Future Values of Time Series</b>, `LH.pred<-predict(fit,n.ahead=8)`
 

******************************************************************************************
 
TIME SERIES PRACTICE

* [R][Previous practice][6]
  * [reference][7]

* [R]Time series beginner
  * [Watch the free videos here FIRST!][8]
    * Often time series are generated as `Xt=(1+pt)Xt−1`, meaning the value of the time series observed at time t equals to the value observed at time t-1, and a small percent change pt at time t. pt is often refered to as <b>return/growth rate</b> of time series, and <b>it should be stable</b>. There is also `Yt=logXt−logXt−1≈pt`. In R, pt is calculated as `plot(diff(log(x)))`
    * <b>Stationary</b>: The <b>mean</b> is constant over time; The <b>correlation structure</b> remains constant over time
    * To convert non-stationary to stationary: Simple differencing can help remove the trend (detrend); You can also use logging + differencing, first logging will tablize the variance, then differencing will do detrend. In R, differencing is using `diff()`, logging is using `log()`. <b>To sum up, logging against Heteroscedasticity, differencing against the trend of the mean</b>.
    * It has been proved that, any stationary time series can be written as a linear combination of white noise, so do ARMA model. Therefore, in R, when you are simulating the model, it uses a list with (p, d, q)
  * [Reference][5]
    * This tutorial is messy, but if you go through it very carefully, there are still many things to learn
  * [My practice code when following the tutorial][4], it's messy too, I use it to try all the methods in the tutorial
  * [My summarized code][9], summarizing methods maybe re-used more often in the future
    * [About AR(0), AR(1) and AR(2)][11]
    * In the definition of AR(p) model [here][10]: the white noise has 0 mean and constant variance
    * As spectral analysis visualization showing, AR model will gradually decrease. MA model will suddenly decrease.
    * What is the role of ARMA, compared with AR, MA?
  * Notes:
    * Different from [my previous practice][6], which was using `ariam` to fit, this is using `sariam` for both fit and forecast. `sariam` is an improvement from `ariam`
    
* [Python]ARIMA Beginner
  * Python ARIMAResults library: http://www.statsmodels.org/devel/generated/statsmodels.tsa.arima_model.ARIMAResults.html
  * For ARIMA, we can follow <b>Box-Jenkins Methodology</b>
    * <b>Model Identification</b>. Use plots and summary statistics to identify trends, seasonality, and autoregression elements to get an idea of the amount of differencing and the size of the lag that will be required.
    * <b>Parameter Estimation</b>. Use a fitting procedure to find the coefficients of the regression model.
    * <b>Model Checking</b>. Use plots and statistical tests of the residual errors to determine the amount and type of temporal structure not captured by the model.
    * download dataset: https://datamarket.com/data/set/22r0/sales-of-shampoo-over-a-three-year-period#!ds=22r0&display=line
    * My code: https://github.com/hanhanwu/Hanhan_Data_Science_Practice/blob/master/sequencial_analysis/python_ARIMA.ipynb
    * Reference: https://machinelearningmastery.com/arima-for-time-series-forecasting-with-python/


******************************************************************************************

READING NOTES

* Outliers Detection for Temporary Data
  * [Find the Book Here][12]
  * [Reading Notes][13]


[1]:https://github.com/hanhanwu/Hanhan_Data_Science_Resources/blob/master/README.md
[2]:https://github.com/hanhanwu/Hanhan_Data_Science_Practice/blob/master/sequencial_analysis/ARMA_example.png
[3]:https://github.com/hanhanwu/Hanhan_Data_Science_Practice/blob/master/sequencial_analysis/ts_r_intro.pdf
[4]:https://github.com/hanhanwu/Hanhan_Data_Science_Practice/blob/master/sequencial_analysis/time_series_beginner.R
[5]:http://www.stat.pitt.edu/stoffer/tsa4/R_toot.htm
[6]:https://github.com/hanhanwu/Hanhan_Data_Science_Practice/blob/master/time_series_predition.R
[7]:https://www.analyticsvidhya.com/blog/2015/12/complete-tutorial-time-series-modeling/?utm_content=buffer529c5&utm_medium=social&utm_source=facebook.com&utm_campaign=buffer
[8]:https://www.datacamp.com/courses/arima-modeling-with-r
[9]:https://github.com/hanhanwu/Hanhan_Data_Science_Practice/blob/master/sequencial_analysis/time_series_beginner_summary.R
[10]:https://en.wikipedia.org/wiki/Autoregressive_model#Definition
[11]:https://en.wikipedia.org/wiki/Autoregressive_model#Graphs_of_AR.28p.29_processes
[12]:https://www.amazon.com/Detection-Temporal-Synthesis-Knowledge-Discovery/dp/1627053751
[13]:https://github.com/hanhanwu/readings/blob/master/ReadingNotes_Temporal_Outlier_Detection.md
