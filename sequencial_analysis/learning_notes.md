# Learning Notes 

## Time Series Data Sets
* [UCI time series datasets][24]
* [CompEngine time series data][25]
* [R-CRAN list of time series packages][26]

## Concepts

* Components of a time series
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
    * stationary series also have constant variance and mean
  * Random Walk models (white noise): the cumulative sum of the zero mean model (a sum of n_i iids at ith iid), and it has 0 mean and constant variace
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
    * It's aiming at seasonal smoothing and gives better estimate of the trend-cycle component
    * rolling by m, then rolling by n again
    * m often choose the periodicity of the seasonal data for seasonal smoothing
    * Check the formula in page 131 of [the book][1], this method will give the time series that are closer to the time index t higher weights, such as t-1, t+1 get higher weights than t-3, t+3. This is different from "recency".
    * This method is a way to make ts stationary
    
* Exponential Smoothing Methods
  * The limitation of moving average and weighted moving average is the ignorance of observation recency effect, and exponential smoothing methods can help deal with this issue by having exponential decaying weights on observations (older data gets lower weights)
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
      * `autolag='AIC'` instructs the function to choose a suitable number of lags for the test by minimize AIC
      * More negative value in ADF statistics represents closer to stationary singal
    * KPSS checks trending stationary
    * And I prefer to check Test Statistic and critical values instead of checking p-value only, since in this way we can get the confidence level of stationary
    * We can also plot the mean, variance, if they are changing a lot, then the ts is not stationary, check [my rolling mean & variance example][15]
  * Differencing methods to convert to stationary
    * First-order differencing: take differences between successive realizations of the time series
      * `x_t - x_t-1`
    * Second-order differencing
      * `(x_t - x_t-1) - (x_t-1 - x_t-2)`
    * Seasonal differencing
      * `x_t - x_t-m`
      * If in the de-trended ts' ACF plot, we are seeing repeated significant autocorrelation (beyond the confidence interval), then use seasonal differencing
      * [An example of seasonal differencing][4], just use `diff()`
    * Weighted moving averages
      * The nxm weighted moving averages method above can help transform ts into stationary
  * Decomposition methods to convert to stationary
    * [Example of decomposing a ts][7]
      * Different from the above decomposition which can be used on the original ts, [Prophet's decomposition comes with the frecasting model][8]
    * Additive model
      * x_t = F_t + S_t + E_t
      * This model is usually applied when thre is a time-dependent trend cycle component but independent seasonality that does not change over time (constant seasonality)
    * Multiplicative model
      * x_t = F_t * S_t * E_t
      * This model often used when there is time-varying seasonality (non-constant seasonality)
    * [Example of applying both additive and multiplicative methods for decomposition, and python built-in `seasonal_decompose`][6]
    
### Auto-Regressive Models
* Besides stationary, exponential smoothing also assumes that random noise is truly random and follows independent identical distribution, but iid assumption often got violated and smoothing is not sufficient to solve the problem. Auto-regressive methods, which will consider the serial correlation between observations can help deal with this.
* AR, MA, ARMA, ARIMA, Seasonal ARIMA all assume stationary
  * Python built-in functions for ARMA, AR, MA will check stationary, if non-stationary will return error; ARIMA and Seasonal ARIMA will use differencing to deal with non-stationary issue
#### AR models
* The way it regress on time series is to regress it with its lag term. So it's good at capturing the trend since it's predicted based on the prior time values
* `p` is the order of AR
  * Check PACF for this, exclude lag 0, the number of significant lags is p
* [Example to create AR model, and forecast on it][13]
  * The residual follows normal ditribution with 0 mean
* <b>Positive atocorrelation is corrected using AR models and negative autocorrelation is corrected using MA models</b>
#### MA models
* It uses the autocorrealtion between residuals to forecast values. This helps adjust for unpredictable events (such as market crash leading to share prices falling that will happen over time)
* `q` is the order for MA
  * Check ACF for q, since it defines error serial correlation well
* [Example to create MA model, and forecast on it][14]
  * The residual follows normal ditribution with 0 mean
#### ARMA models
* The AR(p) models tend to capture the mean reversion effect wheres MA(q) models tend to capture the shock effect in error
* 🌺 Some Thumb rules to determine the orders of ARMA:
  * ACF is exponentially decreasing or forming a sine-wave, and PACF has significant correlation use p
  * ACF has significant autocorrelation and PACF has exponential decay or sine-wave pattern, use q
  * Both ACF, PACF are showing sine-waves, use both p, q
* When there is uncertainty in , p, q values, can try grid search with AIC as the metric, choose the option with the minimum AIC
* After choosing the orders need to check the normality of residuals of the model to see whether it's normally distributed
  * qq-plot, check the [example here][16]
#### ARIMA (Box_Jenkins model)
* Comparing with ARMA model, it added the differencing order `d`, which is used to de-trend the signal to make it stationary before applying ARMA
  * ARIMA(0,1,0) represents a random walk model
  * d represents d-order differencing
* When there is uncertainty in p, d, q values, we can try grid search with AIC as the metric, choose the option with the minimum AIC
* After choosing the orders need to check the normality of residuals of the model to see whether it's normally distributed
  * qq-plot, check the [example here][16]
  * Shapiro-wilk test
#### SARIMA (Seasonal ARIMA)
* SARIMA(p,d,q,m), `m` represents the number of periods per season
* In [this example][16], check the ACF, PACF, at 42 time index it's showing slghtly significant corelation, which may be the seasonality present, so m=42
### Summarize Steps of using (S)ARIMA models 🌺
* Check stationary and residual normality
* Plot ACF, PACF
  * If you will use AR, MA, ARMA need to convert to stationary then plot to decide orders
  * If you use ARIMA, SARIMA, need to plot after differencing the original ts (and it's stationary after differencing) to decide other orders
* If ACF, PACF cannot help decide orders, try grid search and choose the option with minimized AIC
* Check fitted model's residual normality to further valid the model
  
  
## Deep Learning for Time Series Analysis
* Neural networks are suitable in cases when there is little info about the underlying properties such as long-term trend and seasonality or these are too complex to be modeled with traditional models. NN helps extracting complex patterns.
* [All the examples of using MLP, LSTM, GRU, 1D-CNN][17]
  * Notes Summary
    * To use logloss as the metric for regression problem, need to scale the dependent variable into [0,1] range
    * The number of epoches represents the number of times the weight updates. Increasing the number of epochs will reduce the loss but too many epoches will lead to overfitting.
      * Therefore, the number of epochs is controlled by keeping a tap on the loss function computed for the validation set.
    * The network's weights are optimized by the Adam (adaptive moment estimation) optimization.
      * Unlike stochastic gradient descent, Adam uses different learning rates for each weight and seperately updates them as the training progresses
      * The learning rate of a weight is updated based on exponentially weighted moving averages of the weight's gradients and the squared gradients
    * `ModelCheckpoint` tracks the loss function on the validation set and saves the model for the epoch which the loss function has been minumum
* [More advanced methods & examples][18]
### RNN (Recurrent Neural Network)
* When using MLP (multi-layer perceptron), past time steps are fed into the model as uncorrelated independent vaariables, this has ignored the sequential nature of the time series data where observations have correlation with each other. RNN can help in dealing with this.
  * The correlation in a time series can also be interpreted as the memory that the series carries over itself.
* "Bi-directional RNN" uses both forward and backward traversal to improve the ability to capture memory over long ranges
* "Deep RNN": it stacks multiple RNNs on top of each other
* RNN is difficult to train and can suffer from vanishing and exploding gradients that give erratic results during the training
  * Vanishing gradients: The chain of gradient multiplication can be very long. When the multiplication diminishes to 0 and there is no gradient flow from a long-range timestep. Due to the negligibly low values of the gradients, the weights do not update and hence the neurons are said to be saturated
  * Both LSTM, GRU are designed to allow RNN works better in memory transfer for long range sequence
#### LSTM (Long Short Term Memory)
* LSTM introduces 3 new gates, to selectively include the previous memory and the current hidden state that's computed in the same manner as in vanilla RNNs
  * input gate controls the fraction of the newly computed input to keep
  * forget gate determins the effect of the previous timestep
  * output gate controls how much of the internal state to let out
#### GRU (Gated Recurrent Units)
* It has 2 gates
  * update gate determines how much of the previous step's memory is to be retained in the current timestep
  * reset gate controls how to combine the previous memory with the current step's input
  * Comparing with LSTM, it doesn't have the output gate and the internal memory. 
    * The update gate combines the functionality achieved by both input and forget gate in LSTM.  
    * The reset gate combines the effect of the previous memory and the current input, and apply the effect directly to the previous memory
#### LSTM vs GRU
* No one works better than the other in all tasks
* A common rule of thumb:
  * Use GRU when there is less training data
  * Use LSTM for large dataset
#### 1D CNN
* About "convolution"
  * The movement of the filter over the image is "convolution"
  * Multiple convolution layers stacked against each other, generated better features from the original images
* The shape of the convolution layer is (number of samples, number of timestamp, number of features per timestep)
## Recommended Readings
* [Practical Time Series Analysis][1]
* [Time series Q&A][19]
  * Methods and code to deal with missing data
  * How to use Granger Causality test to know if one Time Series is helpful in forecasting another
* [Time series intro][20]
  * Cross correlation: checks whether 2 ts are corrlated with each other. I like the idea of using it in stock, cuz if one tend to drop, the highly correlated one might also drop
    * [Python calculate cross correlation with lag][21], check the highest vote below
* [Sales Uncertainty Prediction][22]
  * Weighted Scaled Pinball loss (SPL) is a metrics used to measure quantile forecasts. This aarticle inclide the implementation

## My Practice
* [ts forecast with basic RNN][23]
  * How to use `Sequential` to build the whole model sequence
  * How to reshape the input for RNN and define the input shape in `Sequential`
  * How to use `ModelCheckpoint` to save the best model and plot the hisory of each epoch training vs validation loss
    * The way it choose the best model is to find the one with the lowest validation loss
  
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
[13]:https://github.com/PacktPublishing/Practical-Time-Series-Analysis/blob/master/Chapter04/Chapter_4_AR.py
[14]:https://github.com/PacktPublishing/Practical-Time-Series-Analysis/blob/master/Chapter04/Chapter_4_MA.py
[15]:https://github.com/hanhanwu/Hanhan_Data_Science_Practice/blob/master/sequencial_analysis/time_series_stationary_measures.ipynb
[16]:https://github.com/PacktPublishing/Practical-Time-Series-Analysis/blob/master/Chapter04/Chapter_4_ARIMA.py
[17]:https://github.com/PacktPublishing/Practical-Time-Series-Analysis/tree/master/Chapter05
[18]:https://github.com/fchollet/deep-learning-with-python-notebooks
[19]:https://www.machinelearningplus.com/time-series/time-series-analysis-python/
[20]:https://www.kaggle.com/janiobachmann/time-series-analysis-an-introductory-start
[21]:https://stackoverflow.com/questions/33171413/cross-correlation-time-lag-correlation-with-pandas
[22]:https://www.kaggle.com/allunia/m5-sales-uncertainty-prediction
[23]:https://github.com/hanhanwu/Hanhan_Data_Science_Practice/blob/master/sequencial_analysis/after_2020_practice/ts_RNN_basics.ipynb
[24]:https://archive.ics.uci.edu/ml/datasets.php?format=&task=other&att=&area=&numAtt=&numIns=&type=ts&sort=nameUp&view=table
[25]:https://www.comp-engine.org/
[26]:https://cran.r-project.org/web/views/TimeSeries.html
