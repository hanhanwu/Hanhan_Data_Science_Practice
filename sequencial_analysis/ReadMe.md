I'm planning to practice more on time series analysis, pattern matching sequential pattern matching and sequence prediction.

## SEQUENCE PREDICTION
* Sequence Prediction - In a word, you get a sequence of behaviors/patterns and try to predict the next.

### CPT For Sequence Prediction
* Problems of LSTMs/RNNs
  * 10s of hours to train, very long time
  * Need to re-train once there is new items that didn't appear in the previous training data
* Compact Prediction Tree (CPT)
  * Much faster than other methods such as Markov Chain, LSTMs/RNNs
  * Official verison [Java]: https://www.philippe-fournier-viger.com/spmf/CPTPlus.php
  * Python version (in development): https://github.com/analyticsvidhya/CPT
  * Data Structure
    * Prediction Tree - it built the training data into a tree, each path can be a sequence
    * Inverted Index - Similar to FP-growth, the dictionary has each node as the key, and the sequences that contain the node as the value. For example:
      * We have Sequence1: A,B,C; Sequence2: B,C; Sequence3: D
      * Then the inverted index will be {A: [Sequence1], B: [Seuqnece1, Sequence2], C: [Sequence1], D: [Sequence3]}
    * Lookup Table - it stores each sequence as the key and the terminal node as the value
  * How it makes prediction
    * Overall View - After building all the 3 data structures with the training data, for each testing sequence, it will predict the mth node that tend to follow this testing sequence. Here by default mth means the last node, however I think it's toally fine for you to define m
    * step 1 - finding common sequences as similiar sequences
      * For each item in a testing sequence, we check the inverted index and find the sequences that contain each of these items. Then we choose the common sequences shared by all these items (intersection). These chosen sequences are the similar sequences.
    * step 2 - generate the consequence for each similar sequence
      * For each similar sequence we got from step 1, we generate the consequence for them
      * A consequence is formed by all the similar sequence items that follow the last item of the testing sequence and removed items that appeared in the testing sequence 
        * If the last item in testing sequence never appeared in the training data, you simpy won't find a prediction
      * For example, we have testing sequence {A,B,C}, similar sequence {A,B,C,D,A,F}, so the consequence will be {D,F} becaue {D,A,F} appeared after C (the last item in the testing sequence), and removed A since it appeared in the testing sequence
    * step 3 - For each consequence, add items in consequence into a shared score_dictionary
      * If the item is in the score_dictionary keys, `score = 1 + (1/number of similar sequences) +(1/number of keys currently in the score_dictionary+1)*0.001`
      * Otherwise, `score = 1 + (1/number of similar sequences) +(1/number of items currently in the score_dictionary+1)*previous_item_score`
    * step 4 - prediction
      * If you predict the most possible n items, just select the n items that with the highest score
      * if you predict the mth item, just select the mth items in score ascending order
  * Reference: https://www.analyticsvidhya.com/blog/2018/04/guide-sequence-prediction-using-compact-prediction-tree-python/?utm_source=feedburner&utm_medium=email&utm_campaign=Feed%3A+AnalyticsVidhya+%28Analytics+Vidhya%29
  * My practice code [Python]: https://github.com/hanhanwu/Hanhan_Data_Science_Practice/blob/master/sequencial_analysis/try_CPT_python.ipynb
    * training data: https://github.com/hanhanwu/Hanhan_Data_Science_Practice/blob/master/sequencial_analysis/train.csv
    * testing data: https://github.com/hanhanwu/Hanhan_Data_Science_Practice/blob/master/sequencial_analysis/test.csv
    * Open Source code: https://github.com/analyticsvidhya/CPT
      * The author allows you to choose the most possible n items, based on this, you can just choose the (N-m)th largest items and choose the smallest one as mth item you want to predict. N is the total number of items in score_dictionary
      * If the last item in testing sequence never appeared in the training data, you simpy won't find a prediction
      * The train and test files muct have the index as the first row. Each record muct have the same length, and must be the same as length as the first row.
  * My Practice code [Java]: https://github.com/hanhanwu/Hanhan_Data_Science_Practice/blob/master/sequencial_analysis/TryCPTPlus.java
    * training data: https://github.com/hanhanwu/Hanhan_Data_Science_Practice/blob/master/sequencial_analysis/training.txt
    * HOW to run the code:
      * Check how to install `SPMF` here: https://github.com/hanhanwu/Hanhan_Data_Science_Practice/tree/master/Frequent_Pattern_Mining
      * create a new folder under `ca.pfv.spmf.test` called "hanhan_test". You can use other folder name, but need to change the path in the .java file above.
      * put both training data and the code into "hanhan_test" folder
      * run the .java file as application
    * Description of `TryCPTPlus.java`
      * The original implementation came from https://github.com/tedgueniche/IPredict
      * Open source is open source.... it has very strict requirements for the input data format:
        * elements in each sequence have to be numbers
        * Each number has to be seperated by " -1 ", and the end of the sequence should be " -1 -2"
          * This will create the limit that your numbers cannot be -1 or -2
      * The testing data is imput by yourself in line 56, 57, 58. In my code, as you can see I input "2,4" to predict which number should follow them
      * One of the bug of this open source code is, if the prediction has more then 1 element that got the same score, the prediction returns empty value....
 
### LSTM For Poem Generation
* As we saw from above that one of the problems of LSTM is, very long running time. In fact I tried it on my own machines, one of my machine has 2GPU and 1TB memory, still running so slow....
* My Python Code: https://github.com/hanhanwu/Hanhan_Data_Science_Practice/blob/master/sequencial_analysis/try_poem_generator.ipynb
  * Method 1 - Character Based
    * The basic idea here is, in the training data, each sequence is 1 character shift from the previous training sequence, the label is the next character of the training sequence. For any testing sequence, assume we want to predict the next n characters (namely, generate poem with next n characters...), when predicting each of these n character, we add predicted characters in the testing sequence to help predict tje next character.
    * In practice, character based tend to use less meory since you have less distinct charactr to store.
    * But even with character based sequence prediction, I only used 10 epoches & 100 batches. Each epoches took 700+ seconds (10+ mins) to run.... That's so slow.
    * Although I have used more complex LSTM, obviously, the number of epoches makes a difference. Look at the final prediction...
  * Method 2 - Word based (Without Tokenization)
    * The way it woks is exactly the same as above character based method. I just changed character to words, and the words are not tokenized, since I expected the result can be more vivid.
    * But obviously, when the number of epoch is small, you simply tend to predict the same word/character each time. LSTM is really time consuming.
    * But also in this case, when using words (even without tokenization), it's faster than character based method, since we will have less training sequences.
* <b>Sample poem input</b>: https://github.com/hanhanwu/Hanhan_Data_Science_Practice/blob/master/sequencial_analysis/sample_sonnets.txt
  
### CPT for Poem Generation
* My Python Code: https://github.com/hanhanwu/Hanhan_Data_Science_Practice/blob/master/sequencial_analysis/CPT_poem_generator.ipynb
  * Finally, I got the most descent poem output from CPT word based method.
  * Of course, how to choose sequence to output a poem, is also an art. In the training data, I used 1000 words in each seuqence, so in the final output, I chose 10 rows, each row is 999 rows away from its next row, in this way each new line we output won't repeat the words from the same sequence (although you can do that, since it's a poem).
* CPT is really much much faster than LSTM. It even finished the task when using the whole sonnects as training data.
* For the large data inout, at the very beginning, I tried to generate poem with selected 7 rows, each row uses 20 characters to predict the next 10 characters. It gave exactly the same output as smaller data sample output.
  * This may indicate that, when the selected testing data is very very small and tend to be unique in the training data, smaller data inout is enough to get the results.
* Then I changed to 12 rows, each row uses 30 characters to predict the next 10 character.
  * The first 5 rows came from continuous rows, which means the next row is 1 character shift from its previous row. If you check their put, although cannot say accurate, but the 5 rows has similar prediction.
* I think CPT can be more accurate when there are repeated sub-sequences appeared in the training data, because the algorithm behind CPT is prediction tree + inverted index + lookup table, similar to FP-growth in transaction prediction, more repeat more accurate.

* Readings
  * Sequence Modeling Use Cases: https://www.analyticsvidhya.com/blog/2018/04/sequence-modelling-an-introduction-with-practical-use-cases/


## TIME SERIES CONCEPTS

### Previous Time Series
* For previous summarized knowledge, check [Time Series section][1]

### Deal with the Missing Data in Time Series
* [How to deal with the missing data in time series, with plot][21]

### Autocorrelation Plot
* In some of my code below, I have used `autocorrelation plot` in multiple places, and the plot was even checked before stationary checking. This is because autocorrelation plot tells how randomly the time series is. Many statistical formula is based on randomness, such as the formula "standard deviation of the sample mean" is only useful when the randomness assumption holds. Meanwhile, most standard statistical tests depend on randomness. The validity of the test conclusions is directly linked to the validity of the randomness assumption.
* The shape of autocorrelation plot and which model to use
  * Exponential, decaying to zero - Autoregressive model. Use the partial autocorrelation plot to identify the order of the autoregressive model.
  * Alternating positive and negative, decaying to zero - Autoregressive model. Use the partial autocorrelation plot to help identify the order.
  * One or more spikes, rest are essentially zero - Moving average model, order identified by where plot becomes zero.
  * Decay, starting after a few lags - Mixed autoregressive and moving average model.
  * All zero or close to zero - Data is essentially random (white noise).
    * Similar to stationary series, white noise is NOT a function of time, and mean, variance do not change over the time.
    * The difference is, white noise is completely random with mean as 0.
  * High values at fixed intervals - Include seasonal autoregressive term.
  * No decay to zero - Series is not stationary.
* References
  * https://www.itl.nist.gov/div898/handbook/eda/section3/autocopl.htm
  * https://www.itl.nist.gov/div898/handbook/pmc/section4/pmc446.htm

### About Stationary
* The first step is often to make the data stationary. <b>A stationary series is one in which the properties – mean, variance and covariance, do not vary with time. It should not show any trend nor seasonality. - Strict Stationary</b>
  * "Trends can result in a varying mean over time, whereas seasonality can result in a changing variance over time, both which define a time series as being non-stationary." This is why when doing time series modeling, we want to make sure the data is stationary, and need to remove the trend and seasonality first.
  * My code - Metrics used to measure stationary: https://github.com/hanhanwu/Hanhan_Data_Science_Practice/blob/7741ccebbbaa3708cd28ddf9c82b6116e3522986/sequencial_analysis/time_series_stationary_measures.ipynb
    * Different metrics may measure different types of stationary, but what we need is strict stationary.
    * When KPSS and ADF are both showing stationary, it tend to be strict stationary.
    * If KPSS is showing (trend) stationary but ADF is not showing (difference) stationary, we can try to remove the trend so that the series may become strict stationary.
      * But also as what I tried in change 1, some differencing method may also make it strict stationary. Does this mean, some differencing may also help remove the trend?
    * If ADF will show (difference) stationary but KPSS shows (trend) stationary, we can try differencing.
  * My code - Methods to make time series stationary: https://github.com/hanhanwu/Hanhan_Data_Science_Practice/blob/master/sequencial_analysis/python_ARIMA.ipynb
  
### ARIMA
* ARIMA is an acronym that stands for <b>AutoRegressive Integrated Moving Average</b>. It is a class of model that captures a suite of <b>different standard temporal structures</b> in time series data.
* AR: Autoregression. A model that uses the dependent relationship between an observation and some number of lagged observations.
* I: Integrated. The use of differencing of raw observations (e.g. subtracting an observation from an observation at the previous time step) in order to make the time series stationary.
* MA: Moving Average. A model that uses the dependency between an observation and a residual error from a moving average model applied to lagged observations.
* Before building the model, there is data preprocessing necessary to make the data stationary. How to check Stationary?
    * First of all, what makes data non-stationary
      * <b>Trend</b>: Varying mean over the time. If you see the data is increasing along the time, off course the mean is increasing too
      * <b>Seasonality</b>: Variance at specific time frame. For example, sales of ice-cream in summer is much more than winter.
    * Why need stationary data
      * I guess it's because, after removing the trend and seasonality, it is easier to make forecast. After the forecast, you can add the trend and seasonality back
    * Check Stationary - Plotting Rolling Statistics
      * Check <b>moving average</b> or <b>moving variance</b>, to see whether they are changing alog the time
    * Check Stationary - Dickey-Fuller Test
      * It has null hypothesis that time series is non-stationary. If <b>Test Statistic is less than Critical Values</b>, then reject the null hypothesis and the data is statinoary.
  * Make time series stationary
    * It's almost impossible to make it perfectly stationary, but we can try to make it closer to be stationary
    * By doing this data preprocessing before forcasting, we remove the trend, seasonality in time series and make it stationary, forcasting can be done on this stationary data, later we can convert forecast values back to the original scale, by adding the trend and seasonality back
    * Mehtods to find trend in order to remove it from time series
      * Aggregation – taking average for a time period like monthly/weekly averages
      * Smoothing – taking rolling averages
      * Polynomial Fitting – fit a regression model
  * For ARIMA, we can follow <b>Box-Jenkins Methodology</b>
    * <b>Model Identification</b>. Use plots and summary statistics to identify trends, seasonality, and autoregression elements to get an idea of the amount of differencing and the size of the lag that will be required.
    * <b>Parameter Estimation</b>. Use a fitting procedure to find the coefficients of the regression model.
    * <b>Model Checking</b>. Use plots and statistical tests of the residual errors to determine the amount and type of temporal structure not captured by the model.
* After data preprocesing for stationary, there could be 2 results:
  * A strictly stationary series with no dependence among the values. This is the easy case wherein we can model the residuals as white noise. But this is very rare.
  * A series with significant dependence among values. In this case we need to use some statistical models like ARIMA to forecast the data.
  * <b>Autocorrelation Function (ACF)</b>: It is a measure of the correlation between the time series with a lagged version of itself. For instance at lag 5, ACF would compare series at time instant ‘t1’…’t2’ with series at instant ‘t1-5’…’t2-5’ (t1-5 and t2 being end points).
  * <b>Partial Autocorrelation Function (PACF)</b>: This measures the correlation between the TS with a lagged version of itself but after eliminating the variations already explained by the intervening comparisons. Eg at lag 5, it will check the correlation but remove the effects already explained by lags 1 to 4.
  * By ploting ACF, PACF, can help us determine the lag at which the correlation is significant, also can disclose the correlation between future data and past data. [Check the example here][19]
    * As you can see in the example, higher bar indicating higher correlation with the lag; slower decay rate indicats higher correlation between future data and current data
* <b>Params (p,d,q)</b> are used to determine which specific ARIMA model will be used
  * <b>p</b>: The number of lag observations included in the model, also called the lag order. <b>It is The lag value where the PACF chart crosses the upper confidence interval for the first time</b>
  * <b>d</b>: The number of times that the raw observations are differenced, also called the degree of differencing.
  * <b>q</b>: The size of the moving average window, also called the order of moving average.
  * When set 0 for a param here, means not using this param. And therefore, we can simply use AR, I or MA model
  * [Summary of rules for identifying ARIMA models][22]
* A linear regression model is constructed including the specified number and type of terms, and the data is prepared by a degree of differencing in order to make it stationary, i.e. to remove trend and seasonal structures that negatively affect the regression model.
* Adopting an ARIMA model for a time series assumes that the underlying process that generated the observations is an ARIMA process.
* [VISUALIZED AR, MA example in ACF, PACF][2], this in fact is a good example. Use ACF to find MA model, use PACF to find AR model
  * [reference][3]
    * It also includes <b>parameter estimation</b>, `fit <- arima(data,order=c(p,d,q))`
    * <b>Diagnostic Checking</b>, to check non-randomness with residuals, `tsdiag(fit)`. The residuals of a “correctly specified” model should be independently distributed, otherwise it's the wrong model
    * <b>Predict Future Values of Time Series</b>, `LH.pred<-predict(fit,n.ahead=8)`

* 11 time series models [python]
  * AR (Autoregression) - it models the next step in the sequence as a linear function of the observations at prior time steps.
    * `AR(p)`, for example when p=1, it means first order AR model
  * MA (Moving Average) - it models the next step in the sequence as a linear function of the residual errors from a mean process at prior time steps.
    * `Ma(q)`, for example when q=1, it means first order MA model
  * ARMA (Autoregression Moving Average) - it models the next step in the sequence as a linear function of the observations and resiudal errors at prior time steps.
    * `ARMA(p,q)`,  p is the order of the autoregressive part and q is the order of the moving average part
  * ARIMA (Autoregressive Integrated Moving Average) - it models the next step in the sequence as a linear function of the differenced observations and residual errors at prior time steps.
    * It combines both Autoregression (AR) and Moving Average (MA) models as well as a differencing pre-processing step of the sequence to make the sequence stationary, called integration (I).
    * `ARIMA(p,d,q)`
    *  It is suitable for univariate time series with trend and without seasonal components.
  * SARIMA (Seasonal Autoregressive Integrated Moving-Average) - it models the next step in the sequence as a linear function of the differenced observations, errors, differenced seasonal observations, and seasonal errors at prior time steps.
    * `model = SARIMAX(data, order=(1, 1, 1), seasonal_order=(1, 1, 1, 1))`
      * The first order is (p,d,q) in ARIMA model, the seasonal_order is (P,D,Q), m at seasonal level, m is the number of time steps in each season.
  * SARIMAX (The Seasonal Autoregressive Integrated Moving-Average with Exogenous Regressors)
    * Exogenous variables are also called covariates and can be thought of as parallel input sequences that have observations at the same time steps as the original series.
    * Sample code:
      * `data1 = [x + random() for x in range(1, 100)]`
      * `data2 = [x + random() for x in range(101, 200)]`
      * `model = SARIMAX(data1, exog=data2, order=(1, 1, 1), seasonal_order=(0, 0, 0, 0))`
      * See data1, data2 here
  * VAR (Vector Autoregression)
    * It is the generalization of AR to multiple parallel time series.
    * `VAR(p)`
  * VARMA (Vector Autoregression Moving-Average)
    *  It is the generalization of ARMA to multiple parallel time series
    * `VARMA(p, q)`
  * VARMAX (Vector Autoregression Moving-Average with Exogenous Regressors)
    * It is a multivariate version of the ARMAX method.
    * `VARMAX(p,q)`, and it also has data1, data2 as above SARIMAX
  * Simple Exponential Smoothing (SES)
    * It models the next time step as an exponentially weighted linear function of observations at prior time steps.
    * It is suitable for univariate time series without trend and seasonal components.
  * HWES (Holt Winter’s Exponential Smoothing)
    * It models the next time step as an exponentially weighted linear function of observations at prior time steps, taking trends and seasonality into account.
    * It is suitable for univariate time series with trend and/or seasonal components.
  * Reference: https://machinelearningmastery.com/time-series-forecasting-methods-in-python-cheat-sheet/
    * It has sample code for each method, all with built-in functions, which is good.
    * Long time ago, I tried DIY 7 methods, but no better than using python built-in methods
      * https://github.com/hanhanwu/Hanhan_Data_Science_Practice/blob/master/sequencial_analysis/time_series_forecasting.ipynb

 
## TIME SERIES PRACTICE

### Suggestions for Time Series Practice
#### Model Validation
* Make sure the validation data happens after training data
* If using cross validation, sklearn has [Time Series Split][20] to make sure testing data happens after training data in each fold
#### Feature Engineering Methods
* Specific time related feature - happens at a certain week of day, time of a day, etc.
* Time based feature - such as which hour, which week, how many minutes used, etc.
* Lag features - Such as lagged count
* Rolling window feature - tend to closder recent time
* Expanding window feature - tend to consider loner historical time
* Domain specific feature
#### [Reference][19]

### [R] [Previous practice][6]
* [reference][7]
 
### [R] Time series beginner
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
    
### [Python] Time Series Methods Cheatsheet
  * My code (7 methods to model seasonality): https://github.com/hanhanwu/Hanhan_Data_Science_Practice/blob/master/sequencial_analysis/time_series_forecasting.ipynb
    * `Holt_Winter` methods works the best, as it takes into account the level data, trend data and seasonal data
    * `Seasonal ARIMA` is an improvement of ARIMA, you need to set the season, you also need to know it's which ARIMA model it is in order to set (p,d,q) in `order()`
    * As you can see, the data used for forecasting is not stationary, but you can still do the forecast
    * How to install the library here
      * If you have laready had `statsmodels`, uninstall it through `pip uninstall statsmodels`
      * `git init`
      * `git clone git://github.com/statsmodels/statsmodels.git`
      * `python setup.py install`
      * `python setup.py build_ext --inplace`
      * Then check whether you can import the library, `from statsmodels.tsa.api import ExponentialSmoothing`
  * reference: 
    * https://www.analyticsvidhya.com/blog/2018/02/time-series-forecasting-methods/?utm_source=feedburner&utm_medium=email&utm_campaign=Feed%3A+AnalyticsVidhya+%28Analytics+Vidhya%29
    * 11 methods: https://machinelearningmastery.com/time-series-forecasting-methods-in-python-cheat-sheet/
    
### [Python] ARIMA Beginner
  * Python ARIMA Results library: http://www.statsmodels.org/devel/generated/statsmodels.tsa.arima_model.ARIMAResults.html
  * download dataset: https://datamarket.com/data/set/22r0/sales-of-shampoo-over-a-three-year-period#!ds=22r0&display=line
  * My code: https://github.com/hanhanwu/Hanhan_Data_Science_Practice/blob/master/sequencial_analysis/python_ARIMA.ipynb
  * Better code: https://github.com/hanhanwu/Hanhan_Data_Science_Practice/blob/master/sequencial_analysis/try_LSTM_Experiment1.ipynb
    * In Checking Stationary stage, used
      * Plotting Rolling Statistics: Compare Test Statistic with Critical Value (1%, 5%, 10%). For example, is Test Statistic is lower than 10% Critical Value, means there is 90% confidence that current data is stationary. Note! When you are checking the visualization, pay attention to y-axis, when it's larger, rolling mean and rolling std lines look flat but in fact may not be better than another graph with smaller y values. So, be careful when compare 2 graphs
      * Dickey-Fuller Test: Check RSS, if you are seeing nan, check whether there is nan or negative value and remove them, before using `np.sqrt`
    * In <b>Making It Stationary stage</b>, used smothooing methods
      * Moving Average - take average of ‘k’ consecutive values depending on the frequency of time series. Drawback is, you have to strictly define time period (such as taking yearly average). <b>[May not be good for data with strong Seasonality]</b>
      * Weighted Moving Average - more recent values are given a higher weight. <b>[May not be good for data with strong Seasonality]</b>
      * Differencing – taking the differece with a particular time lag [Eliminating Trend and Seasonality]
    * After ARIMA forecasting, it also added vack the original sacle, to compare with real observations
  * Check Stationary & Make Data Stationary Reference: https://www.analyticsvidhya.com/blog/2016/02/time-series-forecasting-codes-python/
  * ARIMA Reference: https://machinelearningmastery.com/arima-for-time-series-forecasting-with-python/
  
### ARIMA with grid search, cross validation
  * I was often emotionally against time series practice, because the tutorials I saw were over complex the problem. When there are existing libraries, they implemented their own and the code is not that elegant.... So, finally today, I decided to spend a little bit more time to find an easier solution. Let's just used those existing published libraries, give ourselves an easier life and better solution.
  * My code: https://github.com/hanhanwu/Hanhan_Data_Science_Practice/blob/master/sequencial_analysis/ARIMA_grid_search.ipynb
  
### [Python] RNN - Data Preprocessing for LSTM
  * This is another dataset, the data preprocessing to make data stationary here is better
  * My code: https://github.com/hanhanwu/Hanhan_Data_Science_Practice/blob/master/sequencial_analysis/try_LSTM_Experiment1.ipynb
  
### [R] Stock Prediction - Part 1
  * Dataset Search Library - Quandl: https://www.quandl.com/search?query=
    * Through the website, create an account, then you can search and get access to the data
    * R library: https://cran.r-project.org/web/packages/Quandl/Quandl.pdf
    * US Stock: https://www.quandl.com/product/WIKIP/WIKI/PRICES-Quandl-End-Of-Day-Stocks-Info
      * Just type in US bank names and find results under Wiki EOD Stocks Info
      * Example: https://www.quandl.com/product/WIKIP/WIKI/PRICES-Quandl-End-Of-Day-Stocks-Info
  * My code: https://github.com/hanhanwu/Hanhan_Data_Science_Practice/blob/master/sequencial_analysis/stock_analysis_part1.R
    * First of all, to generate GIF in R, you need `ImageMagick`. To install, check [Create Animated Visualization with R][14], then type `xclock &`. After you are seeing XQuartz is running, then turn on R studio
    * Acording to the code here, didn't have too much excited patterns. The most useful part is, data downloading through Quandl, it's a good data resource
  * reference: https://www.analyticsvidhya.com/blog/2017/09/comparative-stock-analysis/?utm_source=feedburner&utm_medium=email&utm_campaign=Feed%3A+AnalyticsVidhya+%28Analytics+Vidhya%29
  
### [R] Stock Prediction - Part 2
  * <b>Bollinger Bands</b> - The Bands depict the volatility of stock as it increases or decreases. The bands are placed above and below the moving average line of the stocks. <b>The wider the gap between the bands, higher is the degree of volatility.</b>
    * The <b>middle line</b> with N-period moving average (MA); 20-day SMA
    * An <b>upper band</b> at K times an N-period standard deviation above the moving average; 20-day SMA + (20-day standard deviation of price x 2)
    * A <b>lower band</b> at K times an N-period standard deviation below the moving average; 20-day SMA – (20-day standard deviation of price x 2)
    * SMA is Simple Moving Average, Standard Deviation, K and N period is usually set at 20 days. The upper and lower band are placed 2 units above and below respectively.
    * No wonder those people from economic background often check standard deviation with mean....
    ![Bollinger Bands](https://s3-ap-south-1.amazonaws.com/av-blog-media/wp-content/uploads/2017/09/28152914/Bollinger_Band.png)
  * How to read Bollinger Bands
    * If an uptrend is strong, then stock touches the upper band on regular basis and remains above the middle line.This signals the strong movement towards North.
    * If downtrend is strong, then stong touches the lower band on regular basis and remains below the middle line. This signals strong southward movement.
    * During the upward trend the price should not be below the lower band otherwise it is signalling the reverse movement.
    * During the downward trend, if the price move above the upper band then it signals Northward movement.
  * Signals to identify stock behaviours
    * <b>W-Bottom</b>: forms in a downtrend and involves two reaction lows (looks like W), and the second low is lower than the first low but holds above the lower band
    * <b>M-Tops</b>: forms in upper trens and has 2 highs. The first high can be higher or lower than the second high, but does't above the upper band
  * My code: https://github.com/hanhanwu/Hanhan_Data_Science_Practice/blob/master/sequencial_analysis/stock_prediction_part2.R
    * This is the easiest, most good looking stock prediction visualization I have found
    * It uses the combination of regression and ARIMA to do the prediction, predict Close price on the end of the month. This method can be used for other time series prediction
  * reference: https://www.analyticsvidhya.com/blog/2017/10/comparative-stock-market-analysis-in-r-using-quandl-tidyverse-part-i/?utm_source=feedburner&utm_medium=email&utm_campaign=Feed%3A+AnalyticsVidhya+%28Analytics+Vidhya%29

### Facebook Propet
* "Prophet is a procedure for forecasting time series data based on an additive model where non-linear trends are fit with yearly, weekly, and daily seasonality, plus holiday effects. <b>It works best with time series that have strong seasonal effects and several seasons of historical data.</b> Prophet is robust to missing data and shifts in the trend, and typically handles outliers well."
#### Resources
* Examples in R & Python: https://facebook.github.io/prophet/docs/quick_start.html#python-api
* Github: https://github.com/facebook/prophet
* For all the param in `Prophet()`, check https://github.com/facebook/prophet/blob/master/python/fbprophet/forecaster.py
#### About Prophet Forecasting Model
* [Prophet Paper][16]
* `y(t) = g(t) + s(t) + h(t) + εt`
  * g(t): piecewise linear or logistic growth curve for modelling non-periodic changes in time series
  * s(t): periodic changes (e.g. weekly/yearly seasonality)
  * h(t): effects of holidays (user provided) with irregular schedules
  * εt: error term accounts for any unusual changes not accommodated by the model, the parametric assumption here is, `εt` is normally distributed.
* The solution is to frame the forecasting problem as a curve-fitting exercise rather than looking explicitly at the time based dependence of each observation within a time series.
  * A piecewise linear or logistic growth curve trend. Prophet automatically detects changes in trends by selecting changepoints from the data.
    * It seems that it could try multiple types of smoothing methods for seasonality forecasting, and find the optimized one
    * [My code - 7 methods for seasonality modeling (better!)][17], [11 methods for seasonality modeling][18]
  * A yearly seasonal component modeled using Fourier series.
  * A weekly seasonal component using dummy variables.
  * A user-provided list of important holidays.
#### Practice Code
* It seems that Prpphet has at most daily_seasonlity and no hourly, which means it predicts at most at daily level. The major method used here was to calculate hourly fraction (average hourly count/total hourly count)
R and Python versions output similar results, but the digits after the decimal point can be different
  * [Python]https://github.com/hanhanwu/Hanhan_Data_Science_Practice/blob/master/sequencial_analysis/try_prophet.ipynb
  * [R]:https://github.com/hanhanwu/Hanhan_Data_Science_Practice/blob/master/sequencial_analysis/try_prophet_R.R
  
## RNN - LSTM
* <b>LSTM</b>: The Long Short-Term Memory network, or LSTM network, is a recurrent neural network that is trained using Backpropagation Through Time and overcomes the vanishing gradient problem.
  * Instead of neurons, LSTM networks have <b>memory blocks</b> that are connected through layers.
  * A block operates upon an input sequence and each gate within a block uses the sigmoid activation units to control whether they are triggered or not, making the change of state and addition of information flowing through the block conditional.
  * 3 gates within a unit:
    * Forget Gate: conditionally decides what information to throw away from the block.
    * Input Gate: conditionally decides which values from the input to update the memory state.
    * Output Gate: conditionally decides what to output based on input and the memory of the block.
    * The gates of the units have weights that are learned during the training procedure.
* The benefit of LSTM is that, <b>it can learn and remember over long sequences and does not rely on a pre-specified window lagged observation as input</b>. What does this really mean in practice?
  * In Keras, you have to set `stateful=True` when define an LSTM layer. Because by default, Keras maintains the state between data within 1 batch. <b>Between batches, the state will be cleared, by default</b>. Now, if you have `stateful=True`, after the state got cleaned, you can call `reset_states()` to get your states back
    * This also means, you have to manually manage the training process 1 epoch at a time
    * You can reset state at the end of each training epoch, ready for the next training iteration
  * Also, by defualt, data samples will be shuffled within each epoch before being exposed to the network, which will break the state we need for LSTM. So, you need to set `shuffle=False`

* LSTM beginner
  * First of all, I did lots of works to make data stationary here
  * download dataset here: https://datamarket.com/data/set/22u3/international-airline-passengers-monthly-totals-in-thousands-jan-49-dec-60#!ds=22u3&display=line 
  * <b>My code - Experiment 1 (With Stationary)</b>: https://github.com/hanhanwu/Hanhan_Data_Science_Practice/blob/master/sequencial_analysis/try_LSTM_Experiment1.ipynb
    * My godness, each time when there is dimensions problems in neural network, often cost me so much time to figure them out.
    * When using LSTM, it expects input format like this [sample, time_step, dimension], but when you are using Keras LSTM, it doesn't care about the amount of sample, so you are seeing I'm using `model.add(LSTM(4, input_shape=train_X.shape[1:]))`, that is to only use `(time_step, dimension)` to define input_shape
    * Also, pay attention to `Dense(1)` here, because the output should be a single output so that it can compare to `train_Y`, `test_Y`.
    * For more detailed description, check this GitHub answer, `wxs commented on Feb 5, 2016`
    * In the model, I am also using the default `sigmoid` function, This do makes sense in my case. Because my model data input comes from residual, which ranges between [-1,1]. With sigmoid function, it gets [-infinite, +infinite] X value and generates smooth range of values between 0 and 1. Now I think maybe tanh is better because it outputs the results between [-1,1]
    * Check more details for activation fucntions [AI section - Different activation functions][15]
    * At the end of this code, you will see the prediction result using ARIMA forcasting and after using LSTM. Althouh it took me so much effort to make the data stationary and to deal with the data format (when there are Keras neural network, python dataframe and numpy array, things became more complex), the final prediction visualization is difficult for normal people to understand. Customers want to see those predictions make sense, so it's better to have seasonality and trend added back in the forcasting/prediction visualization. In my case, the best stationary data cannot be converted back. Now, let me try Experiment 2, LSTM prediction with seasonality, trend added back
  * <b>My code - Experiment 2 (Without Stationary)</b>: https://github.com/hanhanwu/Hanhan_Data_Science_Practice/blob/master/sequencial_analysis/try_LSTM_Experiment2.ipynb
    * As you can see, without making the data stationary, the final prediction plot makes more sense. Although compared with Experiment1, RMSE in testing data is much larger then the RMSE in training data
    * Method 1 - time t to predict time t+1. You look back 1 time interval, and the time step for LSTM is also 1
    * Method 2 - <b>WINDOW method</b>
      * It allows you to <b>use multiple recent time steps</b> to predict next step. For example, you can use time t-2, t-1, t to predict time t+1, which means you look back 3 time interval but the time step for LSTM is 1
    * Method 3 - Exchange time_step and domension
      * Compared with method 1,2, you just exchange the position of time_step and dimension in `np.reshape`
      * By doing this, you are using previous time_steps to predict t+1 time, instead of phrasing the past observations as separate input features
    * Method 4 - LSTM with memory between batches
      * The LSTM network has memory, which is <b>capable of remembering across long sequences</b>
      * Normally, after each training batch during `model.fit()`, and after`model.predict()` or `model.evaluate()`, the state in network will be reset. In this method 4, you can build state over the entire training sequence and even maintain that state if needed to make predictions.
    * Method 5 - Stacked LSTMs with Memory Between Batches
      * You just stack multiple LSTM models together
      * NOTE: This method requires an LSTM layer prior to each subsequent LSTM layer must return the sequence, so in this previous LSTM model, you set `return_sequences` as True
    * Before method 5, you can see method 1 gave the best results. But when I used method 1 settings in method 5, it gave much worse results than using method 3 settings in method 5. So, in the future, better to try all the methods and see which works better
  * reference: https://machinelearningmastery.com/time-series-prediction-lstm-recurrent-neural-networks-python-keras/
  

* LSTM Univariate Time Series Prediction
  * Univariate vs Multi-variate
    * Univariate predict next 1 step
    * Multi-variate predicts next n steps, which allows larger batch size and faster training
  * My code: https://github.com/hanhanwu/Hanhan_Data_Science_Practice/blob/master/sequencial_analysis/try_LSTM_univariate_time_series.ipynb
    * Base Line Model - Walk Forward Prediction (not LSTM)
      * It is called baseline model does have its reason. You just shift testing data 1 step up and form prediction results, then compare this prediction results with testing data
    * LSTM
      * Step 1 - Data Preprocessing
        * make time series stationary (better to use LSTM without stationary to compare too, sometimes stationary data may not work better)
        * time series transfroms to supervised problem. You can just shift training data down k lags
          * In this case, training & testing vs x & y can be confusing. So, but generating supervised data, we got X, y to fit the model, X are features (here only 1 dimension), y is what to predict. As for training and testing data here are used to evaluate model performance. Both of them have X, y, we just split the data set into first 2 years (training data), lasy year (testing data)
        * make time serles scale
          * To scale data between [-1,1] is because default activation function for LSTM is tanh, which outputs the resuts between [-1,1], so it's better to have input data in this range too
      * Step 2 - Model Fitting & Evaluation
        * `batch_size` has to be a factor of the size of the training and test datasets. In this case, it is 1
        * To help neural network generate reproduciable results, you can:
          * fix seeds for both numpy and tensorflow
          * repeat experiments, and take average of RMSE, to see how well the configuration would be expected to perform on unseen data on average
          * In my code, I used both.
   * reference: https://machinelearningmastery.com/time-series-forecasting-long-short-term-memory-network-python/
     * I really don't like some naming and param settings in it
     
* LSTM Multivariate Time Series Prediction
  * After being tortured by this type of practice in unvariate time series, multivariate appears to be much easier.
  * Compared with univariate, you just need to move multiple features x steps forward
  * my code - one step forward: https://github.com/hanhanwu/Hanhan_Data_Science_Practice/blob/master/sequencial_analysis/multivariate_LSTM_one_step_forward.ipynb
    * In this code, you just need to predict polution, so in fact polution is the only column you need to move forward. But the code here is moving all the original columns forward, and finally drop those unwanted columns, this method has more flexibility in case may want to predict multiple labels
    * In this code, polution just moved 1 step forward
  * my code - multiple steps forward: https://github.com/hanhanwu/Hanhan_Data_Science_Practice/blob/master/sequencial_analysis/multivariate_LSTM_multi_steps_forward.ipynb
    * In this code, it moves 3 hours forward. Check my comments to find the difference between one-step forward
    * It also predicts multiple lables, check my comments
  * reference: https://machinelearningmastery.com/multivariate-time-series-forecasting-lstms-keras/
  
* <b>Summary of Time Series to Supervided Learning</b>
  * According to above LSTM time series predictions, I'd better to summarize major methods used in converting time series to supervised learning. It's in fact very simple. Almost the same for univariate & multivariate, single label & multiple labels.
  * Univariate: your label is your single feature moves x step forward & dropped NA
  * Multivariate, single label: create new columns, each is generated from the relevant original feature & moved x step forward & dropped NA. Then, you only keep the column that you want to predict as label, drop other newly created columns
  * Multivariate, multiple labels: create new columns, each is generated from the relevant original feature & moved x step forward & dropped NA. That's it.
  
## Time Series Prediction
* Data records are recorded in time sequence. With all the sequences, you predict certain classes.
### Time Serirs Predict Human Movement
* Predict whether a person moved or not.
* [Python]Code: https://github.com/hanhanwu/Hanhan_Data_Science_Practice/blob/master/sequencial_analysis/Time_Series_Movement_Prediction.ipynb
  * Each record contains sensors data
  * Each file has to record the same amount of records
  * All files form the training, validation and testing sequences
  * Then you just use LSTM to train the model
  
  
## Channel Attrition

* Find key channels in a sequence

* Channel Attribution Modeling with Markov Chains
  * "An attribution model is the rule, or set of rules, that determines how credit for sales and conversions is assigned to touchpoints in conversion paths." -- From Google
    * For example in this transition disgram
    ![transition diagram](https://github.com/hanhanwu/Hanhan_Data_Science_Practice/blob/master/transion_diagram.png)
    `P(conversion) 
     = P(C1 -> C2 -> C3 -> Conversion) + P(C2 -> C3 -> Conversion) 
     = 0.5*0.5*1*0.6 + 0.5*1*0.6 = 0.45`
  * Markov Chains maps the movement and gives a probability distribution when move from 1 state to another state
    * State space - A possible sets of a process
    * Transition operator - the probability of moving from 1 state to another state
    * Current state probability distribution - probability distribution of being in any one of the states at the start of the process
    * Transition state - In the screenshot above, channels such as C1, C2, C3 are transition states
    * Removel effect - After removing a transition state, `new probability of conversion/probability of conversion`
      * For example, removing C1 from the above, the probability of conversion is 0.5*1*0.6 = 0.3, and the remove effect is 0.3/0.45
    * Transition probability - the probability of moving from one channel to another channel
  * Customer Journey - a sequence of channels, it's a chain of Markov Graph where each vertex is a state, each edge represents transition probability of moving from 1 state to another. It is memory-less Markov Chain, since the probability of reaching to current state only depends on the previous state
  * R ChannelArrition package: https://github.com/hanhanwu/Hanhan_Data_Science_Practice/blob/master/sequencial_analysis/markov_vs_heuristics.pdf
    * I really like the description here. 
    * Heristics Model
      * <b>First Touch Approach</b>: Reward the channel first start the action
      * <b>Last Touch Approach</b>: Reward the last channel made the conversion
      * <b>Linear Approach</b>: Give the same credit to all the channels on the path to the conversion
      * <b>Time Decay Approach</b>: Give subjective weights to each channel on the path to the conversion
    * Markov Model
      * Only care about toal conversion values
      * In first order markov chains, the reaching channel only depends on the previous channel
      * Varied-order Markov Chains (VOM): https://en.wikipedia.org/wiki/Variable-order_Markov_model
  * My code: https://github.com/hanhanwu/Hanhan_Data_Science_Practice/blob/master/sequencial_analysis/markov_chains_beginner.R
    * Get the data here: https://github.com/hanhanwu/Hanhan_Data_Science_Practice/blob/master/sequencial_analysis/Channel_attribution.csv
    * By setting `order` in `markov_model`, you can decide k-order markov chains. Here I'm using fitst order
  * Install packages such as "ChannelAttribution.tar.gz"
    * My goodness, I have never spent longer time to install an R package like this one
    * If you check online, they told you many different methods to install R package when you had failure
    * Now I think, a good starting point is to install from GitHub, if that package has a GitHub version, because the error messages in this install method is more detailed
      * `library(devtools)`
      * `install_github("cran/ChannelAttribution")`
      * Its GitHub: https://github.com/cran/ChannelAttribution
    * In this case, I got error telling me things maybe related to "clang", maybe related to "-lgfortran", I thought mayb ethat clang error caused by "-lgfortran", so I found the solution here:
      * Install -lgfortran: https://thecoatlessprofessor.com/programming/rcpp-rcpparmadillo-and-os-x-mavericks--lgfortran-and--lquadmath-error/
      * For mac Latest OS, download it here: http://gcc.gnu.org/wiki/GFortranBinaries#MacOS
      * Install the .dmg file
      * Then open you terminal, no matter which folder you are you, just type:
        * `mkdir ~/.R`, if you have already had this folder, the terminal you tell you that you had it. If you had it, type `cd ~/.R`, if `Makevars` is there, it's great.
        * Type `cat << EOF >> ~/.R/Makevars`
        * Type `FLIBS=-L/usr/local/gfortran/lib/gcc/x86_64-apple-darwin16/6.3.0 -L/usr/local/gfortran/lib -lgfortran -lquadmath -lm`, change the version name if you downloaded another version
        * Type `EOF`
      * Now in your RStudio, type `library(devtools)`, `install_github("cran/ChannelAttribution")`
    * You may still get the error after install&compile successfully. In this pack, it is important to check source code "LinkingTo"
      * Source: https://www.rdocumentation.org/packages/ChannelAttribution/versions/1.10
      * In LinkingTo, you will see `Rcpp`, `RcppArmadillo`. Install them all and restart your R session
    * Other note
      * Install from local file if you have downloaded the package
        * `install.packages("[zipped package local location]", repos = NULL, type="source")`
  * reference: https://www.analyticsvidhya.com/blog/2018/01/channel-attribution-modeling-using-markov-chains-in-r/?utm_source=feedburner&utm_medium=email&utm_campaign=Feed%3A+AnalyticsVidhya+%28Analytics+Vidhya%29 


******************************************************************************************

READING NOTES

* Outliers Detection for Temporary Data
  * [Find the Book Here][12]
  * [Reading Notes][13]


[1]:https://github.com/hanhanwu/Hanhan_Data_Science_Resources/blob/master/README.md#time-series
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
[14]:https://github.com/hanhanwu/Hanhan_Data_Science_Practice
[15]:https://github.com/hanhanwu/Hanhan_Data_Science_Resources2/blob/master/README.md
[16]:https://github.com/hanhanwu/Hanhan_Data_Science_Practice/blob/master/sequencial_analysis/prophet_paper.pdf
[17]:https://github.com/hanhanwu/Hanhan_Data_Science_Practice/blob/master/sequencial_analysis/time_series_forecasting.ipynb
[18]:https://machinelearningmastery.com/time-series-forecasting-methods-in-python-cheat-sheet/
[19]:https://www.analyticsvidhya.com/blog/2019/12/6-powerful-feature-engineering-techniques-time-series/?utm_source=feedburner&utm_medium=email&utm_campaign=Feed%3A+AnalyticsVidhya+%28Analytics+Vidhya%29
[20]:https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.TimeSeriesSplit.html
[21]:https://www.machinelearningplus.com/time-series/time-series-analysis-python/
[22]:https://people.duke.edu/~rnau/arimrule.htm
