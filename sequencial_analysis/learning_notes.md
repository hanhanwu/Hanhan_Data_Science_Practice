# Learning Notes 

## Time Series Data Sets
* [UCI time series datasets][24]
* [CompEngine time series data][25]
* [R-CRAN list of time series packages][26]

## Forecasting as a Service
* We can learn from these services when building a forecasting system
* [Amazon Forecast][52]
  * Examples: https://github.com/aws-samples/amazon-forecast-samples/tree/master/notebooks

## Statistical Models for Time Series Analysis
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
    
* <b>Statistical Forecasting Models Assumptions</b>
  * Stationary
  * Normally distributed dependent and independent variables
    * Sometimes the skewness of the data is an important info, so make sure you understand that before deciding whether to transoform to normal distribution
    
* Time Series Models
  * Zero Mean models: The model has constant mean and constant variance
    * Observations are assumed to be `iid` (independent and identically distributed), and represent the random noise around the same mean
    * `P(X1=x1, X2=x2, ..., Xn=xn) = f(X1=x1)*f(X2=x2)*...*f(Xn=xn)`
    * stationary series also have constant variance and mean, but it doesn't mean zero mean ts is stationary
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
    * Because it's time independent, it can be reliably used to forecast future time series
    * Positive autocorrelation means the 2 time series moves towards the same direction; negative means opposite directions; 0 means the temporal dependencies is hard to find
    * Autocorrelation is a value between [-1, 1]
  * ACF plot
    * Each vertical bar indicates the autocorrelation between ith ts and i+gth ts, given a confidence level (such as 95%), out of the confidence interval (the threshold lines), the autocorrelation is significant
  * PACF
    * In ACF, the autocorrelation between ith ts and i+gth ts can be affected by i+1th, t+2th, ..., i+g-1th ts too, so PACF removes the influences from these intermediate variables and only checks the autocorrelation between ith ts and i+gth ts
    * Lag0 always has autocorrelation as 1
    * In the example PACF [here][2], besides lag0, only at lag1 there is significant autocorrelation, so the order for AR model is 1
    * Sometimes, PACF has a critical value for a large lag, which is caused by seasonal cycle. We can set `m` in SARIMA with this lag value to catch the seasonality
  * ACF and PACF have the same critical region
    * `[-1.96*sqrt(n), 1.96*sqrt(n)]`
  * The ACF of stationary data should drop to zero quickly. For nonstationary data the value at lag 1 is positive and large.
    * but for white noise (non-stationry), its ACF also drops to zero quickly
  
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
    * This method is a way to make ts stationary through de-trending
    
* Exponential Smoothing Methods
  * The limitation of moving average and weighted moving average is the ignorance of observation recency effect (equally weighted in a window), and exponential smoothing methods can help deal with this issue by having exponential decaying weights on observations (older data gets lower weights)
  * <b>Need to convert the ts to stationary</b> before applying moving average and exponential smoothing methods, since that align with the assumption of these methods
  * Smoothing methods are used to remove random noise, but can be extended for forecasting by ading smoothing factor α, trend factor β, seasonality factor γ in exponential smoothing methods
  * First order exponential smoothing
    * `F_t = α*x_t + (1-α)*F_t-1`, x_i is observation value, α is the smoothing factor in [0,1] range
      * When α=0, the forecasted ts is a constant line, 0 variance
      * When α=1, the forecasted ts is the right shift of the original ts by 1 lag, same variance as the actual ts
      * When α is increasing from 0 to 1, the variance of the forecasted ts is also exponentialy increasing, from 0 to the actial ts' variance
    * [Python implementation of this method][9]
      * The correct way to start exponential smoothing average
  * Second order exponential smoothing
    * `F_t = α*x_t + (1-α)*(F_t-1 + T_t-1)`
    * `T_t = β*(F_t - F_t-1) + (1-β)*T_t-1`
      * β is the trend factor in [0,1] range
      * Second order can capture the variation of the real signal better than first order if the trend component is not constant
    * [Python implementation of this method][11]
      * Similarly, [check DIY smoothing methods here][54] 
    * It's also called as Holt foreacsting
  * Triple order exponential smoothing
    * `F_t = α*(x_t - S_t-L) + (1-α)*(F_t-1 + T_t-1)`
    * `T_t = β*(F_t - F_t-1) + (1-β)*T_t-1`
    * `S_t = γ(x_t - F_t) + (1-γ)S_t-C`
      * γ is the seasonality factor in [0,1] range
    * [Python implementation of this method][12]
      * Similarly, [check DIY smoothing methods here][54] 
    * It's also called as Holt-Winters foreacsting, check how did I use the built-in function [here][10]
  * [How to use python built-in holt, holt-winters functions][55]
    * [My past code][56] 

* Rolling Window vs Expanding Window
  * Rolling window has fixed size, expading window collects all the observations within the specified time range, so the window size of expanding window is not fixed
    
* Methods to Convert to Stationary ts
  * The reason we want to make the ts stationary before applying a model is because, bias & errors, accuracy and metrics will vary over time for non-stationary ts
  * Each stationary testing method has its focus and limitation, not all of them test both mean and variance change along the time, so better to know the limitation to decide which to use
  * I often use both Augumented Dickey-Fuller (ADF) and KPSS to check stationary, [see this example][3]
    * ADF checks differencing stationary, which based on the idea of differencing to transform the original ts
      * It only checks whether the mean of the series changes, as for the change of variance is not formally tested
      * `autolag='AIC'` instructs the function to choose a suitable number of lags for the test by minimize AIC
      * More negative value in ADF statistics represents closer to stationary singal
    * KPSS checks trending stationary
    * And I prefer to check Test Statistic and critical values instead of checking p-value only, since in this way we can get the confidence level of stationary
    * We can also plot the mean, variance, if they are changing a lot, then the ts is not stationary, check [my rolling mean & variance example][15]
  * <b>Differencing is often used to remove trend, log or square root are popular to deal with the changing variance</b>
    * When using log or square root, all the data has to be positive. These methods also make larger values less different and will deemphasize the outliers, need to think whether the transformation is appropriate 
  * Differencing methods to convert to stationary
    * Differencing can not only helps reduce trend and even seasonality, but also could transform the data to a more normally shaped distribution
      * Sometimes a ts needs to be differenced more than once, but differencing much more times may indicate that differencing is not a prefered method
    * First-order differencing: take differences between successive realizations of the time series
      * `x_t - x_t-1`
      * Example of using first-order differencing using `diff()`: https://github.com/PacktPublishing/Practical-Time-Series-Analysis/blob/master/Chapter02/Chapter_2_First_Order_Differencing.ipynb
    * Second-order differencing
      * `(x_t - x_t-1) - (x_t-1 - x_t-2)`
    * Seasonal differencing
      * `x_t - x_t-m`
      * If in the de-trended ts' ACF plot, we are seeing repeated significant autocorrelation (beyond the confidence interval), then use seasonal differencing
      * [An example of seasonal differencing][4], also use `diff()` with seasonal settings
    * Weighted moving averages
      * The nxm weighted moving averages method above can help transform ts into stationary
  * Decomposition methods to convert to stationary
    * [Example of decomposing a ts][7]
      * Different from the above decomposition which can be used on the original ts, [Prophet's decomposition comes with the forecasting model][8]
    * Additive model
      * x_t = F_t + S_t + E_t
      * This model is usually applied when thre is a time-dependent trend cycle component but independent seasonality that does not change over time (constant seasonality)
    * Multiplicative model
      * x_t = F_t * S_t * E_t
      * This model often used when there is time-varying seasonality (non-constant seasonality)
    * [Example of applying both additive and multiplicative methods for decomposition, and python built-in `seasonal_decompose`][6]
    
### Auto-Regressive Models
* Besides stationary, exponential smoothing also assumes that random noise is truly random and follows independent identical distribution, but iid assumption often got violated and smoothing is not sufficient to solve the problem. Auto-regressive methods, which will consider the serial correlation between observations instead
* AR, MA, ARMA, ARIMA, Seasonal ARIMA all assume stationary
  * Python built-in functions for ARMA, AR, MA will check stationary, if non-stationary will return error; ARIMA and Seasonal ARIMA will use differencing to deal with non-stationary issue
#### AR models
* The way it regress on time series is to regress it with its lag term. So it's good at capturing the trend since it's predicted based on the prior time values
* `p` is the order of AR
  * Check PACF for this, exclude lag 0, choose p at those significant lags (if there are multiple significant lags, might need to try multiple, starting with the most significant lag)
* [Example to create AR model, and forecast on it][13]
  * The residual follows normal ditribution with 0 mean
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
  * ARIMA(0,0,0) is white noise
  * ARIMA(0,1,0) represents a random walk model
  * ARIMA(0,1,1) represents first order exponential smoothing model
  * ARIMA(0,2,2) represents second order exponential smoothing (Holt's method), which considers the trend
  * So I think ARIMA(0,3,3) represents Holt-Winters method (Triple order exponential smoothing)
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
  * If you use ARIMA, SARIMA, need to plot after differencing the original ts with d-order (and it's stationary after differencing) to decide other orders, p, q, m
* If ACF, PACF cannot help decide orders, try grid search and choose the option with minimized AIC
  * The orders should not be too large, otherwise may lead to overcomplexed model and overfitting. We should be skeptical if d >= 2, p, q >=5
  * [There is a method in R for automated model selection][37] `auto.arima`
    * It combines unit root tests, minimizing AIC and MLE to obtain an ARIMA model
    * Check all the params here: https://www.rdocumentation.org/packages/forecast/versions/8.13/topics/auto.arima
      * Also supports seasonal model
* Check fitted model's residual normality to further valid the model, [check the bottom of the example here][16]

#### VAR (Vector Autoregression) for multi-variate time series
* All the methods mentioned above are used for univariate ts analysis. Vector Autoregression (VAR) is a multivariate forecasting algorithm that is used when two or more time series influence each other.
  * You need at least two time series (variables)
  * All the time series should influence each other.
    * You can use Portmanteau test to check whether there is multivariate series correlation, and it's interesting that for this type of test, H0 is "there is no serial correlation"
    * I'm also wondering whether VIF will work too?
* [Check details and eaxmple here][38]

## Classical Machine Learning Models for Time Series Analysis
* Instead of making assumptions of the underlying process as statistical models above, these models instead focus on identifying patterns that describe the process’s behavior in ways relevant to predicting the outcome of interest.
* Still need to check the assumptions of the model to see whether need to preprocess the features
### Emsembling models are good choices
* Even though they are not "time-aware" methods
### Clustering Time Series
* We can cluster multiple time series into different clusters, then make forecast for each cluster, assuming each cluster follows a certain behavior that will help the forecasting.
#### To Calculate Cluster Similarity
* Similarity between raw ts data
* Similarity between features
  * ‼️ Make sure the features are the key features that can represent the data, cuz noisy features will only misleading the similarity calculation
* Similarity Calculation Methods
  * [My python code to compare series similarity][57] 
  * Dynamic Time Warping (DTW)
    * It compares the shape of 2 ts, whether the time ranges of the 2 ts are the same DOESN'T matter
      * 2 ts can be warped by being condensed to the same place on the x-axis
    * When comparing the shape, Every point in one time series must be matched with at least one point of the other time series. The first and last indices of each time series must be matched with their counterparts in the other time series. The mapping of points must be forward moving, cannot have backward moving.
    * The cost fuction is often measured as the sum of absolute differences between matched points.
    * [Here's an example][46], but I doubt the use of `linkage = ward`, since it will use euclidean distance which is discouraged in ts similarity calculation
  * Fréchet distance
    * It's like a person walks his dog. They each need to traverse a separate curve from beginning to end, and they can go at different speeds and vary their speeds along the curve so long as they always move in the same direction. The Fréchet distance is the shortest length of leash necessary for them to complete the task following the optimal path.
  * Pearson Correlation
    * 2 ts with same legthen is required.
  * Longest Common Subsequence
    * It finds the length of the longest common subsequence (exactly identical). Similar to DTW, their exact location in the time series is not required to match, and 2 ts don't need to be the same length.
  * ‼️ Avoid Euclidean distance to calculate the similarity between 2 time series, because
    * The displacement along the time axis that isn’t really important for comparison
    * It provides the ability to recognize similarity of magnitudes rather than recognizing the similarity of shapes
  
## Deep Learning for Time Series Analysis
* Neural networks are suitable in cases when there is little info about the underlying properties such as long-term trend and seasonality or these are too complex to be modeled with traditional models. NN helps extracting complex patterns.
* [All the examples of using MLP, LSTM, GRU, 1D-CNN][17]
  * Notes Summary
    * These deep learning models work better when you scale both dependent and independent variables into [-1,1] range ([0,1] range also works)
    * The number of epoches represents the number of times the weight updates. Increasing the number of epochs will reduce the loss but too many epoches will lead to overfitting.
      * Therefore, the number of epochs is controlled by keeping a tap on the loss function computed for the validation set.
    * A good choice to optimize weights is Adam (adaptive moment estimation) optimization.
      * Unlike stochastic gradient descent, Adam uses different learning rates for each weight and seperately updates them as the training progresses
      * The learning rate of a weight is updated based on exponentially weighted moving averages of the weight's gradients and the squared gradients
    * `ModelCheckpoint` tracks the loss function on the validation set and saves the model for the epoch which the loss function has been minimized
* [More advanced methods & examples][18]
### RNN (Recurrent Neural Network)
* TNC data format: `(number of sample per timestep, number of timestep=total samples/number of sample per timestep, number of features)`
* When using MLP (multi-layer perceptron), past time steps are fed into the model as uncorrelated independent vaariables, this has ignored the sequential nature of the time series data where observations have correlation with each other. RNN can help in dealing with this.
  * The correlation in a time series can also be interpreted as the memory that the series carries over itself.
* "Bi-directional RNN" uses both forward and backward traversal to improve the ability to capture memory over long ranges
* "Deep RNN": it stacks multiple RNNs on top of each other
* RNN is difficult to train and can suffer from vanishing and exploding gradients that give erratic results during the training
  * Vanishing & Exploding gradients: It's often the case that gradients would quickly go to zero (not helpful) or to infinity (also not helpful) as it back propagates through time, meaning that backpropagation became difficult or even impossible as the recurrent network was not learning
    * `new weight = weight - learning_rate * gradient`
    * Vanishing gradients: The chain of gradient multiplication can be very long. When the multiplication diminishes to 0 and there is no gradient flow from a long-range timestep. Due to the negligibly low values of the gradients, the weights do not update and hence the neurons are said to be saturated
  * Both LSTM, GRU are designed to allow RNN works better in memory transfer for long range sequence, and to avoid vanishing gradients and reduce exploding gradients, because they tend to keep inputs and outputs from the cell in tractable value ranges, the update gate can learn to pass information through or not, leading to reasonable gradient values.
  
#### LSTM (Long Short Term Memory)
* LSTM introduces 3 new gates, to selectively include the previous memory and the current hidden state that's computed in the same manner as in vanilla RNNs
  * input gate controls the fraction of the newly computed input to keep
    * Event (current input) and STM are combined together so that necessary information that we have recently learned from STM (short-term memory) can be applied to the current input.
  * forget gate forgets information that is not useful
  * cell state: LTM (long-term memory) information that we haven’t forget and STM and Event are combined together in here to update the LTM
  * output gate controls how much of the internal state to let out
    * It uses LTM, STM, and Event to predict the output of the current event which works as an updated STM
  * This video describes the architecture well: https://towardsdatascience.com/illustrated-guide-to-lstms-and-gru-s-a-step-by-step-explanation-44e9eb85bf21
    * It also has a pseudo code
  * The formulas [here][53] are easy to understand
    * "learn gate" --> input gate
    * "forget gate" --> forget gate
    * "remember gate"  --> cell state
    * "use gate" --> output gate
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
#### 1D CNN 💡
* One strategy to combine the speed and lightness of convnets with the order-sensitivity of RNNs is to use a 1D convnet as a preprocessing step before a RNN. 
  * This is especially beneficial when dealing with sequences that are so long that they couldn't realistically be processed with RNNs, e.g. sequences with thousands of steps. The convnet will turn the long input sequence into much shorter (downsampled) sequences of higher-level features. This sequence of extracted features then becomes the input to the RNN part of the network.
  * [See my experiments here][36]
* About "convolution"
  * The movement of the filter over the image is "convolution"
  * Multiple convolution layers stacked against each other, generated better features from the original images
### Deep Learning Platform
* Tensorflow (Google), PyTorch (Facebook), MXNet (Amazon)
* TensorFlow and MXnet are symbolic programming style, while Torch has a more imperative flavor.
  * In a symbolic style of programming, you declare all the relationships up front without having them computed at the time of declaration.
    * Symbolic programming tends to be more efficient because you leave room for the framework to optimize computations rather than perform them right away.
  * In an imperative style of programming, the computation takes place when it is coded, line by line, without waiting to account for what will come later.
    * Imperative programming tends to be easier to debug and understand.
#### [MXNet][47]
* Examples of using MXNet with different RNN models: https://github.com/PracticalTimeSeriesAnalysis/BookRepo/blob/master/Ch10/ForecastingElectricity.ipynb 

  
## Practical Suggestions 🍀
### Be Cautious about "Lookahead" ❣️
* A "lookahead" is the knowledge leaking about the future data. When you need to do model forecasting, lookahead has to be avoided.
### Forecasting profit returns can be more practical on production
* Such as in stock market, forecasting returns is better than forecasting the price
  * And here's a basic example: https://github.com/PracticalTimeSeriesAnalysis/BookRepo/blob/master/Ch14/ForecastingStocks.ipynb

### Time Series Data Storage
#### Relational DB
* [InfluxDB][33] is a time series–specific database. Useful for recording metrics, events, and performing analytics.
* [Prometheus][34] is a "monitoring system and time series database" that works via HTTP. Monitoring first and storage second.
  * It is a pull-based system, which means that the logic for how data is pulled for creating a time series and how often it is centralized can be easily adjusted and inspected.
  * It is also not guaranteed to be entirely up-to-date or accurate, due to its pull-based architecture. Not appropriate for applications where data has to be 100% accurate. 
  * It has a steeper learning curve due to its custom scripting language and the less database-like architecture and API.
#### NoSQL
* Mongo DB is particularly aware of its value as a time series database.
#### Flat File Solution
* Save data in flaat files without any DB at all. The advantages include:
  * System agnostic
  * A flat file format’s I/O overhead is less than that for a database
  * A flat file format encodes the order in which data should be read, whereas not all databases will do so
  * Your data will occupy a much smaller amount of memory than it would on a database because you can maximize compression opportunities
* Only try this when:
  * Your data format is mature
  * Your data processing is I/O bound, so it makes sense to spend development time speeding it up
  * You don’t need random access, but can instead read data sequentially 
* If really want to try this solution, [Xarray][35] is a good choice due to data structure and high performance computing instrument

### Data Preprocessing
#### Thumb of Rules
* Check whether there will be lookahead in each step, especially need to avoid lookahead in forecasting problems
  * Besides checks in each step, also can use cross validation, add each feature slowly and check model performance to see whether a feature will bring in performance jump wihtout a good reason
#### Impute Missing Data
* Missing data can be caused by random problems or systematic problems
  * Random problem example: network crash and suddenly lost a certain amount of data
  * Systematic problem example: when the data came after 10pm, it never got collected
* Main methods to impute missing data
  * With/Without Lookahead
    * Having lookahead can improve imputation quality from domain knowledge sense. However, if the problem is forecasting, lookahead has to be avoided!
  * Forward/Backword fill
    * Forward fill is to carry forward the last known value prior to the missing one
    * Of course, if you need to avoid lookahead, forward fill is the right choice
  * Moving Average
    * If the data is noisy, and you have reason to doubt the value of any individual data point relative to an overall mean, you should use a moving average rather than a forward fill. Since averaging can remove some noise.
    * It doesn't have to be arithmetic average, it can be weighted average or exponential average which give newer observations higher weights
    * However, moving average can reduce the variance of the dataset which might overestimate your model performance. Be cautious!
  * Interpolation
    * "Interpolation is a method of determining the values of missing data points based on geometric constraints regarding how we want the overall data to behave. For example, a linear interpolation constrains the missing data to a linear fit consistent with known neighboring points." Such as LOWESS smoothing method.
      * Linear interpolation accounts for the trend, it works better than moving average in some cases when imputing the missing data based on the trend makes more sense.
    * But these methods may bring in lookahead, look out in forecasting problems
#### Anomaly Detection in Time Series
* [Twitter's AnomalyDetection][51]
  * It's an R package, can be used in both time series data or a vector of numerical data without the timestamp
  * "The underlying algorithm – referred to as Seasonal Hybrid ESD (S-H-ESD) builds upon the Generalized ESD test for detecting anomalies. Note that S-H-ESD can be used to detect both global as well as local anomalies."
#### Downsampling & Upsampling
* Upsampling and downsampling are trying to increasing or decreasing the timestamp frequency, respectively.
* For upsampling, you are just adding more timestamps, not more info
* Pandas `resample()` provides up or down sampling functionality
#### Smoothing
* Smoothing is mainly used to reduce the noise, also strongly related to imputing the missing data
* Moving average, weighted moving average, expoential smoothing, etc.
* First order exponential smoothing
  * Pandas provides exponential weighted moving average method:
    * https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.ewm.html
    * https://pandas.pydata.org/pandas-docs/version/0.17.0/generated/pandas.ewma.html
    * Larger smoothing factor (alpha), it's faster to update the value closer to its current value
    * Because the exponential smoothing method here is build on moving window, can be used to avoid lookahead while normaling the data [like this example][50] 
* Holt's smoothing (second order exponential smoothing)
  * Takes trend into consideration
* Holt-Winters smoothing (third order exponential smoothing)
  * Takes trend & seasonality into consideration
* Kalman filters smooth the data by modeling a time series process as a combination of known dynamics and measurement error. LOESS (short for “locally estimated scatter plot smoothing”) is a nonparametric method of locally smoothing data.
  * Both Kalman and LOWESS will bring in lookahead, so cannot used them for the preprocessing for forecasting problems
  
### Time Series Feature Generation & Feature Selection
#### Feature Generaion with Domain Knowledge
* Especially in stock market (such as those index), healthcare area, etc.
#### Automated Feature Generation 
* ⛔️ Don't overuse auto generation libraries
* Most of these methods need groupby or a series of data, instead of calculating on rolling windows for you
  * The issue of not calculate features with rolling window is, lookahead might appear in the forecasting problem
  * In some cases, calculating features with the whole time series is fine, such as clustering multiple time series using features
* [Python tsfresh][42]
  * "FRESH" stands for feature extraction based on scalable hypothesis tests. 
  * List of features: https://tsfresh.readthedocs.io/en/latest/text/list_of_features.html
  * [My python code][59], I have tried:
    * How to generate features in expanding & rolling window without bringing in lookahead issue
    * How to choose features to generate
    * How to add customized features
  * More examples: https://github.com/blue-yonder/tsfresh/tree/main/notebooks
* [Python Cesium Features][43]
  * List of features: http://cesium-ml.org/docs/feature_table.html
  * Examples: https://github.com/cesium-ml/cesium/tree/master/examples
  * Your can generate features for a list of time series by using `featurize_time_series()`, [see this example][45]
* [Python tsfel][58]
  * It allows you to add new features 
* [R tsfeatures][44]
  * List of features: https://cran.r-project.org/web/packages/tsfeatures/vignettes/tsfeatures.html
    * There are a few functions are designed for rolling windows
#### Feature Selection
* [Python tsfresh][42]
  * "FRESH" stands for feature extraction based on scalable hypothesis tests.
    * The algorithm evaluates the significance of each input feature with respect to a target variable via the computation of a p-value for each feature.
      * Once computed, the per-feature p-values are evaluated together via the Benjamini-Yekutieli procedure, which determines which features to keep based on input parameters about acceptable error rates and the like.
      * An example of calculating feature significance: https://github.com/blue-yonder/tsfresh/blob/main/notebooks/advanced/visualize-benjamini-yekutieli-procedure.ipynb
      * The Benjamini-Yekutieli procedure is a method of limiting the number of false positives discovered during hypothesis testing used to produce the p-values in the initial step of the FRESH algorithm.
* sklearn feature selectors allow you to use different machine learning estimators: https://scikit-learn.org/stable/modules/classes.html#module-sklearn.feature_selection

### Validation, Testing, Evaluation of Time Series Forecasting
#### NOTES 💝
* Exponential smoothing will bring in lookahead in forecasting, because your data has fitted to an exponential smoothing formula before forefasting
  * And any method that will fit both training and testing time series data into a formula. <b>Be cautious about lookahead before applying a preprocessing method</b>
  * With `ewm()`, the exponential smoothing method built on moving window, we can avoid lookahead while normaling the data [like this example][50] 
  * For non time series data, we could preprocess training data first and use the same params to preprocessing testing data, but this method may not work in time series preprocessing
* When there is abnormal dynamics in the data, consider whether to remove them from the data, or build a seperate model for the special dataset or to keep them
#### Cross Validation Formats
* Format 1 - Growing window
  * training: [A,B,C,D], testing:[E,F]
  * training: [A,B,C,D,E,F], testing:[G,H]
  * training: [A,B,C,D,E,F,G,H], testing:[I,J]
  * [Iplemented in sklearn TimeseriesSplit()][48]
* Format 2 - Moving window (backtest)
  * training: [A,B,C,D], testing:[E,F]
  * training: [B,C,D,E], testing:[F,G]
  * training: [C,D,E,F], testing:[G,H]
  * stride >=1
* Format 3 - Expanding widow
  * Similar tomoving window, but instead of having fixed window size, it specifies the starting & ending points of a window, any observation between the time range will be included, so the window size varies
  * This method might be better at reducing overfitting than movig window solution
* Format 4 - Forwardtest (Walk Forward)
  * training: [A,B,C,D], testing:[E,F]
  * training: [G,H,I,J], testing:[K,L]
  * training: [M,N,O,P], testing:[Q,R]
#### What to Check for Further Model Improvement
* Performance metrics
* Plot the output of your predictions, to see whether the output makes sense
* Plot the residuals of the model over time
  * If the residuals are not homogenous over time, your model is underspecified
* Test your model against a simple temporally aware null model (baseline)
  * A common null model is that every forecast for time t should be the value at time t – 1, a constant line. Worse than the null model, your model is useless
* Study how your model handles outliers
  * Sometimes the ability of forecasting outliers will lead to overfitting and need to ignore the outliers; sometimes your task is to detect the outliers and the model needs to detect them
* Conduct temporal sensitivity analysis
  * Whether the model treats similar termoral patterns in similar ways? comparing in both training and testing sets
* Open to more
#### Validation Gotchas
* "lookahead"
* Structural Change
  * The change may lead to the model change, and we can monitor the changes in a forecasting system
  * [Detect Structural Changes in Time Series-2017 paper][49]
    * How to test structural change with hypothesis
      * R `strucchange` package supports both cumulative SUMS (CUSUMS) and moving SUMS (MOSUMS): https://cran.r-project.org/web/packages/strucchange/strucchange.pdf
      * Python I found OLS-CUSUMS: https://www.statsmodels.org/stable/generated/statsmodels.stats.diagnostic.recursive_olsresiduals.html
    * Detect breakpoints of the structural change
      * R `strucchange` package also provides the `breakpoints()` method
    * Without the built-in method, we can also check the change significance

### Spurious Correlation
* When 2 series are showing high correlation it may not be real correlation. First of all need to think whether the correlation makes sense
* There is no fixed way to determine whether the 2 series are truly correlated, here're the methods to check
  * Method 1 - check the correlation between differenced series of the 2 ts, if no longer showing correlation, then likely that the original 2 series are not truly correlated to each other
  * Method 2 - check the correlation between diff(series1) and lag(diff(series2)), if no longer showing correlation, then likely that the original 2 series are not truly correlated to each other
* Often times it’s the relationship between data at different points or the change over time that is most informative about how your data behaves.

### Data Simulation
* SimPy, a process-based discrete-event simulation framework: https://simpy.readthedocs.io/en/latest/
* Simulation with Markov Chain Monte Carlo (MCMC) method
  * "The basis of a Monte Carlo simulation is that the probability of varying outcomes cannot be determined because of random variable interference. Therefore, a Monte Carlo simulation focuses on constantly repeating random samples to achieve certain results. A Monte Carlo simulation takes the variable that has uncertainty and assigns it a random value. The model is then run and a result is provided. This process is repeated again and again while assigning the variable in question with many different values. Once the simulation is complete, the results are averaged together to provide an estimate."
  * "A Monte Carlo simulation will help you figure out what a particular distribution or series of terms looks like, but not how those terms should evolve over time. This is where a Markov chain comes in. It calculates a probability of transitioning between states, and when we factor that in, we take “steps” rather than simply calculating a global integral."
    * In a Markov process, the probability of a transition to a state in the future depends only on the present state (not on past information).
    * [An example of MCMCsimulation][32]
* Deep Learning Simulation
  * Can capture complex and nonlinear dynamics, hard to leak privacy since it's a blackbox
  * The drawback is, even the practitioners don't understand the dynamics of the system...

### Other Knowledge
* Psychological Time Discounting: People tend to be more optimistic (and less realistic) when making estimates or assessments that are more “distant” from us.
#### State Space Models
* It doesn't assume stationary on the data and allows time-varying coefficients instead fixed coefficients. However, also because of this, there can be many parameters to set, which could lead to overfitting, computationally expensive and hard to understand.
* The Kalman Filter
  * Predict the current value merely based on the last value. This method is more useful when the internal dynamics of the system are very well understood, such as the moving trace of a rocket (you know how to calculate with Newton's Law)
* Hidden Markov Models (HMM)
  * It is a rare instance of unsupervised learning in time series analysis, meaning there is no labeled correct answer against which to train. "Markov process" means it is “memoryless” that the probabilities of future events can be fully calculated given only the system’s current state, no need to know earlier states. 
  * A Hidden Markov Model represents the same kind of system, except that we are not able to directly infer the state of the system from our observations. For example, in a time series, it has states of A, B, C, D, 4 subseries, but you are only seeing 1 series.
  * Baum-Welch algorithm is used to identify the distinct emission probabilities for each possible hidden state and identify the transition probabilities from each possible hidden state to each other possible hidden state.
    * To find the optimal sequence of hidden states, because the complexity grows exponentially with the number of time steps. So need to use EM (expectation–maximization) algorithm. But EM cannot guarantee to find the global optimal, so better to try multiple different initialization.
  * Viterbi algorithm is used to identify the most likely hidden state for each time step given the full history of observations.
    * The Viterbi algorithm searches all possible paths that could explain a given observed time series, where a path indicates which state was occupied at each time step, so it's guaranteed to find the best solution.
    * It is a dynamic programming algorithm designed to fully and efficiently explore the range of possible fits by saving the solutions to portions of a path so that as the path is lengthened there’s no need to recompute all possible paths for all path lengths
      * Dynamic programming is also described as a technique of memoization, which saves smaller problems' solutions in order to solve a bigger problem faster
  * HMMs will perform better on longer time series with more opportunities to observe/infer state transitions.
  * [Python example of using HMM][39]
    * Params: https://ogrisel.github.io/scikit-learn.org/sklearn-tutorial/modules/generated/sklearn.hmm.GaussianHMM.html
* Bayesian structural time series (BSTS)
  * For more details, check [python PyDLM][40] and [its user manual][41]


## My Practice
* [ts forecast with basic RNN][23]
  * Using tensorflow2.3
  * [Same code with tensorflow 2.4][29]
    * Main change is in the library import
  * The input data shape for RNN is `(number of timestamp, number of features)`
  * How to use `Sequential` to build the whole model sequence
  * How to reshape the input for RNN and define the input shape in `Sequential`
  * How to use `ModelCheckpoint` to save the best model and plot the hisory of each epoch training vs validation loss
    * The way it choose the best model is to find the one with the lowest validation loss
* [ts forecast with basic LSTM, GRU][30]
* [ts forecast with Stacking & Bidirectional RNN][31]
  * Having tried different activation functions, y was scaled into [0,1] range, but `tanh` works better than `sigmoid` and all better than `relu`. 
    * There is also an online suggestion that, `tanh` is a better choice for regression problem, while `relu` may not be
  * `optimizer=Adam(amsgrad=True, learning_rate=0.1)`, I have chosen adam optimizer
    * `amsgrad=True` is trying to make Adam converge
  * Besides stacking all with LSTM, also tried to stack with LSTM & GRU, in this case, mised models didn't improve the performance
  * Although don't think bidirectional work for many industry forecasting problems, still gave it a try. In this case, it got the same performance as stacking method with linear activation
* [ts forecast with 1D CNN & LSTM][36]
  * 1D CNN only is faster when there is same amount of epoches and batch_size
    * In this case, both experiments got same performance whe epoch and batch_size were the same
  * Looks like performed better than above stacked RNNs, but stacked method was using learning_rate=0.1 while here was using learning_rate=0.001 by default.
    

## Recommended Readings
* [Practical Time Series Analysis][1]
  * Its code: https://github.com/PacktPublishing/Practical-Time-Series-Analysis
* [Practical Time Series with more details][27]
  * Its code: https://github.com/PracticalTimeSeriesAnalysis/BookRepo
  * This books covers a lot but very basic, lots onf concepts are not clear to understand, the GitHub code really sucks
* [Time series Q&A][19]
  * Methods and code to deal with missing data
    * Backward Fill
    * Linear Interpolation
    * Quadratic interpolation
    * Mean of nearest neighbors
    * Mean of seasonal couterparts
    * Make sure to impute without lookahead in forecasting problems, although this will drop the imputatio quality, it's still better than making your forecasting model too good to be true
  * 💡 How to use Granger Causality test to know if one Time Series is helpful in forecasting another 
* [More methods to test stationary other than ADF, KPSS][28]
* [Time series intro][20]
  * Cross correlation: checks whether 2 ts are corrlated with each other. I like the idea of using it in stock, cuz if one tend to drop, the highly correlated one might also drop. But we need to pay attention to spurious correlation.
    * [Python calculate cross correlation with lag][21], check the highest vote below
    * Maybe we can also check the correlation between differenced ts in order to avoid spurious correlation
* [Sales Uncertainty Prediction][22]
  * Weighted Scaled Pinball loss (SPL) is a metrics used to measure quantile forecasts. This article includes the implementation
  
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
[27]:https://github.com/PracticalTimeSeriesAnalysis/BookRepo
[28]:https://perma.cc/D3F2-TATY
[29]:https://github.com/hanhanwu/Hanhan_Data_Science_Practice/blob/master/sequencial_analysis/after_2020_practice/ts_RNN_basics_tf2.4.ipynb
[30]:https://github.com/hanhanwu/Hanhan_Data_Science_Practice/blob/master/sequencial_analysis/after_2020_practice/basic_lstm_gru.ipynb
[31]:https://github.com/hanhanwu/Hanhan_Data_Science_Practice/blob/master/sequencial_analysis/after_2020_practice/advanced_RNN.ipynb
[32]:https://github.com/PracticalTimeSeriesAnalysis/BookRepo/blob/master/Ch04/Ising.ipynb
[33]:https://github.com/influxdata/influxdb
[34]:https://github.com/prometheus/prometheus
[35]:https://github.com/pydata/xarray/blob/master/doc/index.rst
[36]:https://github.com/hanhanwu/Hanhan_Data_Science_Practice/blob/master/sequencial_analysis/after_2020_practice/ts_1DCNN.ipynb
[37]:https://perma.cc/P92B-6QXR
[38]:https://www.machinelearningplus.com/time-series/vector-autoregression-examples-python/
[39]:https://github.com/PracticalTimeSeriesAnalysis/BookRepo/blob/master/Ch07/HMM.ipynb
[40]:https://github.com/wwrechard/pydlm
[41]:https://pydlm.github.io/#dynamic-linear-models-user-manual
[42]:https://github.com/blue-yonder/tsfresh
[43]:https://github.com/cesium-ml/cesium
[44]:https://github.com/robjhyndman/tsfeatures
[45]:https://github.com/PracticalTimeSeriesAnalysis/BookRepo/blob/master/Ch09/Classification.ipynb
[46]:https://github.com/PracticalTimeSeriesAnalysis/BookRepo/blob/master/Ch09/Clustering.ipynb
[47]:https://github.com/apache/incubator-mxnet/tree/master/example
[48]:https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.TimeSeriesSplit.html
[49]:https://github.com/hanhanwu/readings/blob/master/Structural_Change_in_Economic_Time_Series.pdf
[50]:https://github.com/PracticalTimeSeriesAnalysis/BookRepo/blob/master/Ch14/ForecastingStocks.ipynb
[51]:https://github.com/twitter/AnomalyDetection
[52]:https://docs.aws.amazon.com/forecast/latest/dg/what-is-forecast.html
[53]:https://www.analyticsvidhya.com/blog/2021/01/understanding-architecture-of-lstm/?utm_source=feedburner&utm_medium=email&utm_campaign=Feed%3A+AnalyticsVidhya+%28Analytics+Vidhya%29
[54]:https://medium.com/@srv96/smoothing-techniques-for-time-series-data-91cccfd008a2#f381
[55]:https://www.statsmodels.org/stable/examples/notebooks/generated/exponential_smoothing.html
[56]:https://github.com/hanhanwu/Hanhan_Data_Science_Practice/blob/master/sequencial_analysis/time_series_forecasting.ipynb
[57]:https://github.com/hanhanwu/Hanhan_Data_Science_Practice/blob/master/sequencial_analysis/after_2020_practice/series_similarity.ipynb
[58]:https://github.com/fraunhoferportugal/tsfel
[59]:https://github.com/hanhanwu/Hanhan_Data_Science_Practice/blob/master/sequencial_analysis/after_2020_practice/try_tsfresh.ipynb
