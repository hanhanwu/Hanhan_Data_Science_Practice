
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
  * In this chapter, it's majorly about CDF, another distribution funtion, which maps values to percentile. Compared with PMF in Chapter 2, PMF maps values to probability.
  * The code here is using 3 methods to plot percentile and the count. Although percentile may be more reliable to do binning, in the case here when comparing the distribution of pregency length between first born and other born, small equal bin size (step method) plots better results than customized bins
  * It also proves python random methods tend to be uniform distribution
    * Python generate uniform distributed random numbers: `random.uniform(min,max)`
    * Python generate random values from a list with replacement: `random.choice(lst)`
    * Python generate random values from a list without replacement: `random.sample(lst, num_of_values)`

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
  * CDF (Cumulative Distribution Function): maps value to their percentile rank in the distribution.
    * For both discrete and continuous values, a value can be mapped to percentile. In this way, even later you need to do binning, it can be easier and more reliable. Otherwise, when you are decide bin sizes and number of bins, you won't know whether the data can distribute well
    *  The binning here is to divide percentile values into bins
    * Percentile indicates how many values (in percentage) are no more than the current value. For example, if a number ranked at 90% percentile, it means there are 90% values no more than this number
  * For percentile, we also know IQR = 75% Percentile - 25% Percentile = Q3-Q1
    * Normally, outliers are below Q1-1.5* IQR, or higher than Q4+1.5*IQR. But in practice, 1.5 may remove more useful data. You need to decide this value based on the real situation
  * Knowing the distribution, you can also do data sampling
    

[1]:https://github.com/hanhanwu/Hanhan_Data_Science_Practice/blob/master/Applied_Statistics/thinkstats_chapter1.ipynb
[2]:https://github.com/hanhanwu/Hanhan_Data_Science_Practice/blob/master/Applied_Statistics/thinkstats_chapter2.ipynb
[3]:https://github.com/hanhanwu/Hanhan_Data_Science_Practice/blob/master/Applied_Statistics/thinkstats_chapter3.ipynb
