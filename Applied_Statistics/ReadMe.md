## Statistics Tables
* [Z table - Having z score to get p-value][8]
  * I found this one contains more z-values to check.
  * NOTE, as we can see `P(Z ≤ z-score)` here all on left side, which means the probability that we obtained is to the left of the Z-score. However, p value is to the right-hand side of the threshold, with the fact that the total area under the normal Z distribution is 1, `p-value = 1-P(Z ≤ z-score)`
  * When to use:
    * `z = (Population Mean - Sample Mean)/(Population std/sqrt(sample records))`
    * If we can get the z-score, with Z table, we can get the relevant p-value.
  * Details about p-value, check [here][10]
* [Online z-score to p-value][9]
  * Different from the table above, this one will diretly give you the right side probability - p-value


## Experiments

### Maximum Likelihood Estimation
* With Maximum Likelihood, you can estimate population parameters from sample data such that the probability (likelihood) of obtaining the observed data is maximized
* It seems that all the likelihood function can be written in this way regardless of the distribution, and the purpose of maximum of likelihood is to estimate the coefficients `β0, β1`, in R it's `θ0, θ1`, so that the estimated coefficients will maximize the likelihood function.
<img src="https://github.com/hanhanwu/Hanhan_Data_Science_Practice/blob/master/Applied_Statistics/images/likelihood_function.png" width="180" height="60">

* My code [R]: https://github.com/hanhanwu/Hanhan_Data_Science_Practice/blob/master/Applied_Statistics/try_maximum_likelihood_estimation.R
  * Download the data: https://s3-ap-south-1.amazonaws.com/av-blog-media/wp-content/uploads/2018/07/Train_Tickets.csv
  * The data distribution can be treated as <b>Poisson Distribution</b>
    * Poisson Distribution is a discrete probability distribution that expresses the probability of a given number of events occurring in a fixed interval of time or space if these events occur with a known constant rate and independently of the time since the last event.
  * The way to evaluate the problem is to calculate `µ` of Poisson Distribution or you can understand it as the average of `y` (what you need to predict, in this case it's `Count`). 
  * So in order to calculate `µ`, you need to calculate the coefficients vector `θ`. You can form a linear model:
    * `µi = xi'* θ`, such that `µ = θ0 + x1*θ1 + pow(x2,2)*θ2 + .... + pow(xn,n)*θn`
    * To allow negative values, you can also have `log(µi) = xi'* θ`, or `µi = exp(xi'* θ)`
  * In this code, we have `x` - elapsed_weeks, `y` - Count, `µ = exp(θ0 + x*θ1)`, so we can calculate negative likelihood = `-sum(y*(log(µ)) - µ)`
    * Poisson Distribution has `L(θ;x) = Pr{Y = y|µ} = exp(-µ) * pow(µ, y)/y!`
    * Likelihood `LL(θ;x) = log(L(θ;x)) = sum(y*(log(µ)) - µ)`
    * Using negative likelihood so that the optimization will become minumization, same as maximize positive likelihood
  * Method 1 - DIY model, and you calculate coefficients vector using R `mle`
    * With the calculated coefficients, you can calculate µ, consider it as the average of Count, and evaluate with testing data Count, to get RMSE
    * To use `mle`, you won't find `stats4` package in installing tool, you can simply run `library(stats4)`
  * Method 2 - R `glm` 
    * It will calculate the coefficients for you when you define the distribution in `family`
    * In the final evaluation part, you can see when I was using `exp()` for prediction results and comparing with the observations, the error was smaller. Maybe this is because `glm` was using `log()` function as default link function for poisson distribution and you need to convert back in prediction results, https://www.statmethods.net/advstats/glm.html
* Python also have a library similar to R `glm` - `pymc`
* Inspirations
  * Pay attention to the power of taking log
  * To deal with Giant data computation
    * We know there is map-reduce, parallel computing, streaming to deal with large amount of data computation. But sometimes none of them could help when there are more strict requirements came from customers...
    * So when dealing with hiant dtaa computation, instead of using all the data at once, how about using central limit theorem and maximum likelihood estimation
      * With central limit theorem, you generate many fixed size samples (record>=30), only compute on each sample, and generate the mean of each sample. When the distribution of these mean formed the normal distribution, your samples are representative, and you can just use the `µ` of the normal distribution as the average of your population results
        * With this method, you don't need to check any other distribution except the distribution of all the mean from the samples
      * But if you even cannot generate that large amount if samples, how about calculate maximum likelihood.
        * This need you to know the distribution of data, and what you need to predict
        * Then you need to know y (what you need to predict), x (features used for the prediction), formula between x,µ and θ, likelihood formula. Then you do the optimization work in order to calculate µ
        * Basicly, you can treat a finalized maximum likelihood model as a linear regression model
* Reference: https://www.analyticsvidhya.com/blog/2018/07/introductory-guide-maximum-likelihood-estimation-case-study-r/?utm_source=feedburner&utm_medium=email&utm_campaign=Feed%3A+AnalyticsVidhya+%28Analytics+Vidhya%29
  * It's lack of many things that a tutorial should have.... you even don't know how did he get those columns and plots.... Better to check my code.
  

## Interesting Statistics Methods

### 8 Simple Sampling Methods
* Probability Sampling
  * Random Sampling, such as monte carlo method
  * Systematic Sampling - The first individual is selected randomly and others are selected using a fixed sampling interval
    * This interval can be N/n, N is population size while n is sample size
    * It might also lead to bias if there is an underlying pattern in the selected items
  * Stratified Sampling
  * Clustering Sampling - Randomly group the population into clusters, then choose the whole cluster as a sample
* Non-probability Sampling
  * Convenience Sampling - Individuals are selected based on their availability and willingness to take part
    * tend to lead to bias
  * Quota Sampling - choose items based on predetermined characteristics of the population
    * such as choose odd numbers
  * Judgement Sampling - experts decide whom to participate
  * Snowball Sampling - randomly select the initial group, then each individual in the group to norminate more participates


## Resources

* Think Stats
  * Download the book here: http://greenteapress.com/thinkstats/
  * [The book with my reading marks][6]
  
* Graphical Analytics: https://www.itl.nist.gov/div898/handbook/eda/section3/eda33.htm
  * Its explain, sample, questions a plot can answer, etc. are all pretty good!

* Greek Alphabet: https://en.wikipedia.org/wiki/Greek_alphabet


## Data Source

* 2002 Preganent Data: ftp://ftp.cdc.gov/pub/Health_Statistics/NCHS/Datasets/NSFG/2002FemPreg.dat
  * Data Dictionary: ftp://ftp.cdc.gov/pub/Health_Statistics/NCHS/Datasets/NSFG/stata/2002FemPreg.dct
* 2002 Respondent Data: /Health_Statistics/NCHS/Datasets/NSFG/2002FemResp.dat
* About NSFG Survey: https://www.icpsr.umich.edu/nsfg6/
* My code - Read Raw Data: https://github.com/hanhanwu/Hanhan_Data_Science_Practice/blob/master/Applied_Statistics/read_nsfg_data.py
  * preg output: https://github.com/hanhanwu/Hanhan_Data_Science_Practice/blob/master/Applied_Statistics/2002FemPregOut.csv
    * <b>birthorder</b> is empty when it's not live birth...
    * <b>finalwgt</b> is the statistical weight associated with the respondent. It is a floating-point value that indicates the number of people in the U.S. population this respondent represents. Members of oversampled groups have lower weights.
  * resp output: https://github.com/hanhanwu/Hanhan_Data_Science_Practice/blob/master/Applied_Statistics/2002FemRespOut.csv

* Other Data Sources
  * WalframAlpha: http://www.wolframalpha.com/
  

## Think Stats Practice Code

* [My Code - Chapter 1][1]
* [My Code - Chapter 2][2]
  * In this chapter, the author only mentioned Count & Probability to create histograms. In my code, you will see 4 types:
    * Type 1 - Count & Probability for Discrete variable
    * Type 2 - Equal Bin Size, Count & Probability for Continuous variable
    * Type 3 - Normalized Count for Discrete variable
    * Type 4 - Equal Bin Size, Normalized Count for Continuous variable
* [My Code - Chapter 3][3]
  * In this chapter, it's majorly about CDF, another distribution funtion, which maps values to percentile. Compared with CDF in Chapter 2, PMF maps values to probability.
  * The code here is using 3 methods to plot percentile and the count. Although percentile may be more reliable to do binning, in the case here when comparing the distribution of pregency length between first born and other born, very small equal bin size (step method) plots better results than customized bins
    * Here you may get confused why I can use `step()` frunction to plot CDF because CDF should generate continuous distribution, which is the opposite to step distribution. But the step function here is using very small bins which finally formed a continuous looks
  * It also proves python random methods tend to be uniform distribution
    * Python generate uniform distributed random numbers: `random.uniform(min,max)`
    * Python generate random values from a list with replacement: `random.choice(lst)`
    * Python generate random values from a list without replacement: `random.sample(lst, num_of_values)`
* [My Code - Chapter 4][4]
  * Python methods to generate random values based on different CDF continuous function
  * Plot different CDF
  * Mehtods to tell whether a certain dataset satisfy a certain distribution
  * Knowing CDF continuous function formula, we can also generate random numbers based on a certain distribution,
but it's easier to use pyton built-in functions
* [My Code - Chapter 9][5]
  * Before you calculate any correlation, better to plot scatterplot to see whether there is linear or non-linear correlation. If it's non-linear, methods like pearson correlation won't work
  * When the dataset you used has been rounded off and you may lose some info, you can try `jittering` to add random noise in the data, in order to reverse the effect caused by data rounded off
  * For large amount of data, you can try hexbin to plot, it divides the graph into hexagonal bins and color each bin according to how many data points fall into it

* [Decay Rate & Decay Curve - My Code][7]
  * We can use this time of curve to represent the decrease of the population along with the time increasing, or an exponential distribution with x increase y decreases
  * In this code it has:
    * mean time decay curve generation, given `x,y`. Python will fit the curve and generate `a,b,k` for you
    * half life decay curbe generation, given `x,y`. Python will fit the curve and generate `a,b,k` for you
    * Given y at mean_time or half_life, calculate the x at the point and calculate decay rate
  * NOTE: Sometimes, the initial value can be terribly huge and much larger than all the other values. This may lead to a negative decay rate. The solution is, either you rmeove this outlier for both x and y, or when you are generating x, y values, user higher resolution for x (time) and record y values. For example, you were using x for each 1 hour, now you change to each second


## Learning Notes

* Mean describes the central tendency, while Variance describes the spread
* `Variance = sum(sqrt(Xi-mean))/n, i = 1,...,n`
  * `Xi-mean` is the deviation from the mean, so variance is mean squared deviation
  * People use standard deviation because it can more meaningful than variance. Variance us using squared unit (such as squared pund) but standard deviation is using unit (such as pound)
* HDI (Highest Density Interval) - The HDI summarizes the distribution by specifying an interval that spans most of the distribution, say 95% of it, such that every point inside the interval has higher believability than any point outside the interval.
  * When using HDI, we often check within HDI (say 95%), whether the valeus are all lower than a certain thereshold x.
* Relative Risk - a ratio of 2 probabilities
  * For example, first born babies have 20% probability to be born late, other babies have 15%. Relative Risk will be 1.3 here. So first born babies have 30% more likely to be late
* Conditional Probability
  * We all know it means the probability under a certain condition, but I didn't think too much about its practical use
    * For example, it's 38th week now, and you want to know the probability of a born to be born next week (39th week). To calculate this probability, you can just check 39th week to 45th week numbers, and calculate the probability among them, instead of calculating the probability among 1st week to 45th week. I know, it sounds so simple....
* PMF vs CDF
  * PMF (Probability Mass Function): maps values to probabilities in representing the distribution.
    * For discrete values, you count each value, and map the count to probability
    * For continuous values, you define bins, and drop each value into a bin, finally map the bin drop count to probability
      * The binning here is to divide continuous values into bins
  * CDF (Cumulative Distribution Function): maps value to their percentile rank in the distribution, the function is a <b>continuous function</b> instead of a step function.
    * For both discrete and continuous values, a value can be mapped to percentile. In this way, even later you need to do binning, it can be easier and more reliable. Otherwise, when you are decide bin sizes and number of bins, you won't know whether the data can distribute well
    *  The binning here is to divide percentile values into bins, then plot with a <b>continuous function</b>
    * Percentile indicates how many values (in percentage) are no more than the current value. For example, if a number ranked at 90% percentile, it means there are 90% values no more than this number
  * For percentile, we also know IQR = 75% Percentile - 25% Percentile = Q3-Q1
    * Normally, outliers are below Q1-1.5* IQR, or higher than Q4+1.5*IQR. But in practice, 1.5 may remove more useful data. You need to decide this value based on the real situation
  * Knowing the distribution, you can also do data sampling
* Different CDF Distributions
  * <b>Exponential Distribution</b>
    * `CDF(x) = 1- exp(-lambda*x)`
      * lambda decides the distribution shape
      * `mean=1/lambda`, is the mean of the distribution
      * `median=ln(2)/lambda`, is the median of the distribution
    * This method often used in time series. <b>x-axis means the time interval between events</b>, y-axis is the percentile of the count of events that happened in a time interval.
      * For example, x-axis show time interval between 77 and 99 minutes, y-axis shows the percentile of the count of, how many events happened in 77 mins time interval with each of other events, how many events happened in 7 mins time interval with each of other events, etc.
      * <b>If all the events are equally likely to occur at any time, the distribution should be exponential distribution</b>, x is the time interval, y is the event counts at each time interval
    * `CCDF = exp(-lambda*x)`, is the complementary of CDF
    * Some plot looks like exponential distribution, but may not be. <b>To prove whether it is exponential distribution</b>, you can take natural log of y and see whether you can get a straight line, because `log(y)=-lambda*x`
  * <b>Pareto Distributon</b>
    * Power Law distribution, 80-20 distribution
    * Originally, the economist Pareto used this distribution to describe 80 percent wealthy were held in 20% population. Later this has been used in many phenomenons to describe majority something are owned by a small percentage population
    * `CDF(x) = 1- pow(x/xm, -alpha)`, xm is the minimum possible value
    * `CCDF(x) = pow(x/xm, -alpha)`
  * <b>Normal Distribution</b>
    * CDF with normal distribution
    ![cdf with normal distribution](https://github.com/hanhanwu/Hanhan_Data_Science_Practice/blob/master/Applied_Statistics/cdf_normal.png)
      * μ is the mean, σ is the standard deviation
  * <b>Lognormal Distribution</b>
    * log each value from a notmal distribution
* Frequentism vs Bayesianism
  * Frequentism are people who defines probability based on frequency, but for things dind't happen before, they put the probability as 0, which won't work well in random case or unpredictable case. Bayesianism are people who will give whatever event a probability, but they can be subjective.
* Let's review some probability formulas
  * `P(A and B) = P(A)*P(B)`, only when A, B are independent
  * `P(A|B) = P(B|A) = 0`, mutally exclusive
  * `P(A or B) = P(A) + P(B)`, either event
  * In general, `P(A or B) = P(A) + P(B) - P(A and B)`
  * `P(H|E) = P(H)*P(E|H)/P(E)`, Bayes's theorem
    * It describes how the probability of a hypothesis gets updated over time
    * P(H) is prior probability, P(H|E) is posterior
    * P(E|H) is the likelihood of the evidence, means the probability of the evidence conditioned on a hypothesis
    * P(E) is normalizing constant, used to normalize the result to a probability
* Monty Hall - In a word, you choose switch has 2/3 probability, you choose stick has 1/3 probability, because the host knows where is the car. 
* Poincare - Well, what I have learned is, don't sell bread to a statistician
* <b>Compare variability - coefficient of variation</b>
  * Conefficient of Variation = `σ/μ`. When 2 groups have different mean, you can use this value to compare the variance
* Binomial Distribution
  * For example, in each trail, you choose a value between 0 and 100, and you want to know the chance of getting k 7s in n trails
  * Binomial Distribution `PMF(k) = binomial_coefficient * pow(p,k) * pow((1-p), n-k)`
    * Binomial Coefficient is read as "n choose k" = `n!/(k!*(n-k)!)`
    * n is total number of trails, k is the number of event you want to track, p is the probability of the event in each trail
* Monte Carlo
  * Calculate probability by simulating random process
  * Strength: easy and fast to write a simulation without knowing too much about probability
  * Weakness: When calculating rare event probability, it can take very long time
* Texas Sharpshooter Fallacy is an informal fallacy which is committed when differences in data are ignored, but similarities are stressed
* Skewness - measures how asymmetric the distribution is
  * "Skew Left" means the distribution extends farther to left than right
    * Extreme values have more effect on mean, so when a distribution skews left, its mean is less than median
  * <b>Pearson's median skewness coefficient</b> also measures skewness
    * `3*(mean-median)/σ`, σ is standard deviation
    * It's robust which means, it's less vulnerable to the outliers
* Convolution - An operation that computes the distribution of the sum of values from 2 distributions
* Central Limit Theorem
  * If we add values drawn from normal distribution, the distribution of the sum is also normal
  * If we add up values from other distributions, the sum does not have continuous distributionsmentioned above
  * However, if we add up large number of values from any distribution, the distribution of the sum will converges to normal
  * Contral Limit Theorem - If the distribution of the values has mean and standard deviation (μ, σ), the distribution of the sum is close to N(n*μ，n*pow(σ,2))
  * Limitations
    * Values have to be drawn independently
    * Values have to come from the same distribution (this requirement can be relaxed)
    * The distribution need to have finite mean and standard deviation, so distributions such as Pareto won't work
    * The number of values you need before seeing convergence depends on the skewness of the distribtion
* PMF, CDF, PDF
  * PMF represents a DISCRETE set of values. From PMF to CDF, you just compute the cumulative sum. From CDF to PMF, you calculate the differences in cumulative probabilities
    * In python, cumulative sum of a list is `numpy.cumsum(lst)`
  * PDF is the derivative of CDF, CDF is the integral of PDF.
    * PDF maps values to probability density, not probability. To get probability, you have to integrate
  * If you divide a PDF into a set of bins, you can generate a PMF that is the approximation of the PDF -- a technique used in Bayesian estimation
* Illusory Superiority: The tendency of people to imagine that they are better than the average
* <b>Hypothesis Testing</b>
  * I was always wondering why did people chose those ways to define null hypothesis, and then how to calculate that p-value. Today I got the answer from this book.
    * Alternative Hypothesis indicates an apparent effect will happen by chance. And therefore, null hypothesis indicates the apparent effect is not happened by chance. 
    * Statistically Significant means an apparent effect is unlikely to happen by chance. When p-value is lower than threshold α, we accept null hypothesis, saying the apparent effect is statistically significant, which means it unlikely to happen by chance
    * To calculate p-value, for example, you want to know how significant the DIFFERENCE between 2 groups is.
      * You have 2 groups, Group1 has n observations while Group2 has m observations
      * Calculate mean differnce between Group1 and Group2 as `δ`
      * Then from each group, you choose same sized sample to form a pair. Generate 1000 sample pairs. For each sample pair you compute the difference in mean, and count how many mean differences in these 1000 sample pairs are >= δ. Such as now you have 177 mean difference >= δ, we can say p-value=0.177.
      * You also have a threshold `α`. Such as α=18%, p-value is lower than this threshold, so we accept null hypothesis and it means the DIFFERENCE between the 2 groups is not by chance.
      * In common practice, we choose α=10%, 5%, 1%
  * Threshold α is also the probability of false positive (accepted null hypothesis while it's false)
    * To decrease threshold, we can reduce the false positive, but it may also increase false negative (reject null hypothesis when it's true)
    * The best way to decrease both false positive and false negative is, increase sample size
  * Test Statistic - measures total deviation, deviation means the difference between observed value and expected value. In a word, it's error function. Chi-square is a common choice for this measure
  * The author also mentioned Bayesian probability and likelihood ratio
* Estimation
  * Estimation is a process of inferring the params of a distribution from a sample.
    * This sounds abstract. I like the way that the author describes in the book, for each exmaple, it starts with an estimation game.
    * With estimation, you can estimate single value for a distribution from a sample - Point Estimate
    * With estimation, you can also estimate an interval that contains true values for a distribution, and you also give the probability of this interval to tell how many times the true values will be in this interval among all the samples - Confidence Intervals
  * Bayesian Estimation
    * With this method, you can calculate the interval that contains the true value, with the given probability
    * Basic Implementation Steps
      * Make Uniform Suite - You have been given the upper and lower bound `[a,b]` for the value you need to estimate. With this step, you divide this range into equal sized bins
        * The final confidence interval you wil get is narrower than this range
        * You define the bin size yourself
        * This is uniform because all the bins have the same prior probability
      * Calculate Likelihood
        * This should be the core, difference distributions have different likelihood calculation methods
      * Update & Normalize
        * With Prior probability and Likelihood, you can update posterio and normalize, just like Bayesian theorem
      * With the given probability, such as 90%, you can use 5% & 95% percentiles as the confidence interval
        * percentage = 90%
        * `prob = (1-percentage)/2, confidence interval = [prob, 1-prob]`
      * The author implemented 2 examples:
        * Exponential Distribution: http://greenteapress.com/thinkstats/estimate.py
        * Locomotive Problem: http://greenteapress.com/thinkstats/locomotive.py
          * If there is enough evidence to swamp the priors, then all estimators tend to converge
     * With Censored Data 
       * When x is in a given condition
       * Just replace your likelihood calculation method with given condition
         * Such as for exponential distribution, your likelihood function is PDF, with censored data, just give the PDF a range of x
* Correlation - The relationship between variables
  * <b>Correlation is NOT Causation</b>, sounds simple to remember, but how many people never make this type of mistake?
  * The challenges to measure correlation between 2 variables are: they have different units, and they may have different distributions
    * Solution 1 - Transform all values into standard scores (Pearson Coefficent of correlation)
      * `standard score z = (x-μ)/σ`, in this way, Z becomes dimensionless (no unit) and the distribution has 0 mean and variance 1
      * If X is normally distributed, so does Z. If X is skewed or has outliers, so does Z. This is why Pearson Coefficient is not robust and only works well when the data is close to normal distribution. Using percentile ranks can be more robust
    * Solution 2 - Transform values to percentile ranks (Spearman Coefficient)
  * Covariance - Measures the tendency of 2 variables to vary together
    * Deviation `dXi = Xi-μ`
    * The product of deviations `dXi* dYi` will be positive when X, Y deviations have the same sign; negative when the deviations have the opposite sign
    * Covariance `Cov(X, Y) = sum(dXi*dYi)/n`
    * But covariance is difficult to interpret because its unit is the product of X unit and Y unit
  * Pearson Correlation
    * To solve the interpretatio problem caused by covariance, here comes correlation
    * `pi = ((Xi - μx)/σx) * ((Yi - μy)/σy)`, the product of X, Y standard scores
    * Pearson Correlation `ρ = sum(pi)/n = Cov(X, Y)/(σx*σy)`
      * Its value is in range [-1,1]
      * ρ=1 or -1 both indicate perfect correlation (from one variable can predict the other), one means positive correlation, the other is negative correlation
      * <b>But Pearson Correlation only measures linear correlation, and the data should be roughly normally distributed, and not robust to outliers</b>
      * For non-linear correlation, ρ can still be 0. So, a good practice is to plot scatter plot before calculating any correlation
      * Correlation is NOT related to slope
   * Spearman Correlation
     * Spearman's rank correlation is an alternative that mitigates the effect of outliers and skew distributions
       * First of all, you compute the rank of each value, or <b>apply a transform that makes the data more nearly normal</b>
         * For example, if the data is close to lognormal, you can take the log of each value
       * Then compute Pearson Correlation with the transformed data
   * Linear Least Squares Fit
     * Correlation measures the relationship between 2 variables, but NOT the slope. Linear Least Squares Fit can help
     * "Linear Fit" is a line intended to model the relationship between variables; "Least Squares" minimize the mean squared error (MSE) between the line and the data
       * Assume this line is `α + βx`
       * Residual `εi = (α + βxi) - yi`
     * Minimize the sum of squared residuals, `min(sum(pow(εi, 2)))`
   * Goodness of fit - You want to know how good a linear model is, one way is to measure its predictive power
     * Coefficient of Determination (R Square)
       * `R Square = 1 - Var(ε)/Var(Y)`
         * When you didn't know anything, you guess y_. `Var(Y) = pow((y_ - yi),2)/n`
         * When you know the function for this linear model, you have `Var(ε) = pow((α + βxi - yi), 2)/n`
         * e.g. When R Square is 0.79, it means it has reduces the MSE of your precitions by 79%
       * For Linear Least Squares model, `R Square = pow(ρ,2)`
* Interpretability vs Accuracy
![Interpretability vs Accuracy](https://github.com/hanhanwu/Hanhan_Data_Science_Practice/blob/master/Applied_Statistics/interpretability_vs_accuracy.png)
  * Algorithms with higher flexibility also leads to higher variance, but lcan be ower bias; those with lower flexibility have lower variance, but also could have higher bias

[1]:https://github.com/hanhanwu/Hanhan_Data_Science_Practice/blob/master/Applied_Statistics/thinkstats_chapter1.ipynb
[2]:https://github.com/hanhanwu/Hanhan_Data_Science_Practice/blob/master/Applied_Statistics/thinkstats_chapter2.ipynb
[3]:https://github.com/hanhanwu/Hanhan_Data_Science_Practice/blob/master/Applied_Statistics/thinkstats_chapter3.ipynb
[4]:https://github.com/hanhanwu/Hanhan_Data_Science_Practice/blob/master/Applied_Statistics/thinkstats_chapter4.ipynb
[5]:https://github.com/hanhanwu/Hanhan_Data_Science_Practice/blob/master/Applied_Statistics/thinkstats_chapter9.ipynb
[6]:https://github.com/hanhanwu/readings/blob/master/thinkstats.pdf
[7]:https://github.com/hanhanwu/Hanhan_Data_Science_Practice/blob/master/Applied_Statistics/decay_rate.ipynb
[8]:http://pages.stat.wisc.edu/~ifischer/Statistical_Tables/Z-distribution.pdf
[9]:https://www.socscistatistics.com/pvalues/normaldistribution.aspx
[10]:https://github.com/hanhanwu/Hanhan_Data_Science_Practice/blob/master/Applied_Statistics/Learning_Notes/p_value.md
