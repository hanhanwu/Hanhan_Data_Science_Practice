
<b>Think Stats</b>
* Download the book here: http://greenteapress.com/thinkstats/

********************************************************************************

<b>Data Source</b>

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
  

********************************************************************************

<b>Practice Code</b>

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


********************************************************************************

<b>Learning Notes</b>

* Mean describes the central tendency, while Variance describes the spread
* `Variance = sum(sqrt(Xi-mean))/n, i = 1,...,n`
  * `Xi-mean` is the deviation from the mean, so variance is mean squared deviation
  * People use standard deviation because it can more meaningful than variance. Variance us using squared unit (such as squared pund) but standard deviation is using unit (such as pound)
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

[1]:https://github.com/hanhanwu/Hanhan_Data_Science_Practice/blob/master/Applied_Statistics/thinkstats_chapter1.ipynb
[2]:https://github.com/hanhanwu/Hanhan_Data_Science_Practice/blob/master/Applied_Statistics/thinkstats_chapter2.ipynb
[3]:https://github.com/hanhanwu/Hanhan_Data_Science_Practice/blob/master/Applied_Statistics/thinkstats_chapter3.ipynb
[4]:https://github.com/hanhanwu/Hanhan_Data_Science_Practice/blob/master/Applied_Statistics/thinkstats_chapter4.ipynb
