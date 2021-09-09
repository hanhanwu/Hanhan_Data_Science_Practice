# A/B Test Experiments

### Suggestions
* [A/B test suggestions][2]
  * I found some of the suggestions here are helpful, especially questions in determing your experiment goal.
  * The terminology description here is also good.
* [Ultimate guide in A/B testing][3]
  * The suggestions in Don'ts and Do's
  * Ideas about what usually to test
  * Some A/B test cases
* [Some Suggestions in multivariate testing][4]

* [Frequentist Methods vs Bayesian Method in A/B Test][5]
  * Besides using the tool called Exploratory for Bayesian A/B test, [here's the theoritical method for Bayesian hypothesis test][6]

## Thorough Experiment with Confidence Interval, p-value for A/B test
* This is a typical A/B test problem. 2 Options to decide whether the change on web elements will make a difference
### [My code][1]
* Online calculators can be used
  * Estimate sample size: https://www.evanmiller.org/ab-testing/sample-size.html
    * Given Type 1 error rate α, type II error rate β, basedline conversion rate, min detectable effect.
  * Estimate p-value for binomial distribution: https://www.graphpad.com/quickcalcs/binomial1/
    * Given number of success, total trails, probability of success in biomial distribution.
#### Overall Process
1. Choose invariant metrics and evaluation metrics.
2. Estimation of baselines and Sample Size
  * Estimate population metrics baseline
  * Estimate sample metrics baseline
  * Estimate sample size for each evaluation metrics, only keep the metrics that have practical amount of sample size.
3. Verify null hypothese - Control Group vs. Experiment Group
  * Sanity check with invariant metrics
  * Differences in evaluation metrics, using error of margin & confidence interval, Dmin (min change that's significant to the business) to check both statistical, practical significance.
  * Differences in trending (such as daily trending), using p_value to check statistical significance.

### [A simplified A/B test process][7]
1. null hypothesis
    * 2 options, no difference
2. control group & test group (new option)
    * sampling method, sample size —> reduce bias, polulation representative
3. data collection 
    * what to collect: daily conversion rates for both groups  —> sample size = days
    * testing period: 2 months
    * what to test: difference bt avg. daily conversion rate, across the test period
4. statistical significance
    * to be statistical significant —> the difference bt avg. is not due to error or random chance
    * 2 sample t-test
      * Significance level (alpha): The significance level, also denoted as alpha or α, is the probability of rejecting the null hypothesis when it is true. Generally, we use the significance value of 0.05
      * P-Value: It is the probability that the difference between the two values is just because of random chance. P-value is evidence against the null hypothesis. The smaller the p-value stronger the chances to reject the H0. For the significance level of 0.05, if the p-value is lesser than it hence we can reject the null hypothesis
      * Confidence interval: The confidence interval is an observed range in which a given percentage of test outcomes fall. We manually select our desired confidence level at the beginning of our test. Generally, we take a 95% confidence interval
    * p_value < alpha: reject H0
    * [Besides t-test, when to use other test types][9]
* [Some suggestions to comapre the control and the test groups][8]
  * Such as Mann Whitney U-test, a non-parametric method used to compare the 2 groups when the distribution is not normal

[1]: https://github.com/hanhanwu/Hanhan_Data_Science_Practice/blob/master/Applied_Statistics/ABTest_Experiments/detailed_ABTest_Experiment.ipynb
[2]:https://github.com/dwyl/learn-ab-and-multivariate-testing
[3]:https://www.smashingmagazine.com/2010/06/the-ultimate-guide-to-a-b-testing/
[4]:https://www.smashingmagazine.com/2011/04/multivariate-testing-101-a-scientific-method-of-optimizing-design/
[5]:https://github.com/hanhanwu/Hanhan_Data_Science_Practice/blob/master/Applied_Statistics/Learning_Notes/AB_Test.md
[6]:https://github.com/hanhanwu/Hanhan_Data_Science_Practice/blob/master/Applied_Statistics/Learning_Notes/bayesian_analysis/Bayesian_Approaches_for%20Hypothesis_Tests.md
[7]:https://www.analyticsvidhya.com/blog/2020/10/ab-testing-data-science/?utm_source=feedburner&utm_medium=email&utm_campaign=Feed%3A+AnalyticsVidhya+%28Analytics+Vidhya%29
[8]:https://www.analyticsvidhya.com/blog/2021/03/a-b-testing-measurement-frameworks%e2%80%8a-%e2%80%8aevery-data-scientist-should-know/?utm_source=feedburner&utm_medium=email&utm_campaign=Feed%3A+AnalyticsVidhya+%28Analytics+Vidhya%29
[9]:https://www.analyticsvidhya.com/blog/2021/09/hypothesis-testing-in-machine-learning-everything-you-need-to-know/?utm_source=feedburner&utm_medium=email&utm_campaign=Feed%3A+AnalyticsVidhya+%28Analytics+Vidhya%29
