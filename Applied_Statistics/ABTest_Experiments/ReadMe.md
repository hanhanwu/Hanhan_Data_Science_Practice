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

## Thorough Experiment with Confidence Interval, p-value for A/B test
* This is a typical A/B test problem.
### [My code][1]
* Online calculators can be used
  * Estimate sample size: https://www.evanmiller.org/ab-testing/sample-size.html
    * Given Type 1 error rate α, type II error rate β, basedline conversion rate, min detectable effect.
  * Estimate p-value for binomial distribution: https://www.graphpad.com/quickcalcs/binomial1/
    * Given number of success, total trails, probability of success in biomial distribution.
### Overall Process
1. Choose invariant metrics and evaluation metrics.
2. Estimation of baselines and Sample Size
  * Estimate population metrics baseline
  * Estimate sample metrics baseline
  * Estimate sample size for each evaluation metrics, only keep the metrics that have practical amount of sample size.
3. Verify null hypothese - Control Group vs. Experiment Group
  * Sanity check with invariant metrics
  * Differences in evaluation metrics, using error of margin & confidence interval, Dmin (min change that's significant to the business) to check both statistical, practical significance.
  * Differences in trending (such as daily trending), using p_value to check statistical significance.


[1]: https://github.com/hanhanwu/Hanhan_Data_Science_Practice/blob/master/Applied_Statistics/ABTest_Experiments/detailed_ABTest_Experiment.ipynb
[2]:https://github.com/dwyl/learn-ab-and-multivariate-testing
[3]:https://www.smashingmagazine.com/2010/06/the-ultimate-guide-to-a-b-testing/
[4]:https://www.smashingmagazine.com/2011/04/multivariate-testing-101-a-scientific-method-of-optimizing-design/
