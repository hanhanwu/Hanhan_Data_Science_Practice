# p-value 🔮


💖 I have read many about applying p-value in data work, but the more I read, the more cofusing it could be. Finally today found a well structured and clear tutorial [here][1], have to take notes here. 💖

## What is p-value
p-value is the "total probability" of getting any value to the right hand side of the red point, when the values are picked randomly from the population distribution. 

In a word, right side space represents this probability. Threfore, larger p-value is, the higher probability that the value was selected from the population distribution.
<p align="center">
  <img width="300" height="200" src="https://github.com/hanhanwu/Hanhan_Data_Science_Practice/blob/master/Applied_Statistics/Learning_Notes/images/p-value.png">
</p>

## p-value and alpha

* Alpha is the threshold, 1-alpha is the significant level.
* Null Hypothesis: Sample distribution is similar to the population distribution. (Or the sample was obtained from the population.)
* p-value > alpha
  * Fail to reject NULL Hypothesis
  * Not significant.
  * The sample results are just a low probable event of the population distribution and are very likely to be obtained by luck.
* p-value < alpha
  * The result is NOT in favor of NULL Hypothesis
  * Significant
  * The results obtained from the sample is an extremity of the population distribution (an extremely rare event), and hence there is a good chance it may belong to some other distribution.
<p align="center">
  <img width="300" height="200" src="https://github.com/hanhanwu/Hanhan_Data_Science_Practice/blob/master/Applied_Statistics/Learning_Notes/images/p_value_alpha.png">
</p>

## Steps to Apply p-value in Real World
* Create NULL Hypothesis and Alternative Hypothesis
* Similarity test (such as calculating z-score)
  * `z = (Population Mean - Sample Mean)/(Population std/sqrt(sample records))`
* Calculate p-value based on the above step
* Compare p-value and alpha, and interpret the result
* [The example here is good][1]

## p-value in machine learning output
* It can be used in feature selection.
  * NULL Hypothesis: The independent variable has no significant effect over the target variable.
* In the output of `statsmodels`, you will see Adjusted R Square as the performance metric, and `P > |t|` which represents p-value. You can try to remove those features have largest p-values that are larger than an alpha value, observe whether model performance will improve.
  * When p-value is larger than alpha here, fail to reject NULL hypothesis, which means the feature has no significant effect over the target.




[1]:https://www.analyticsvidhya.com/blog/2019/09/everything-know-about-p-value-from-scratch-data-science/?utm_source=feedburner&utm_medium=email&utm_campaign=Feed%3A+AnalyticsVidhya+%28Analytics+Vidhya%29
