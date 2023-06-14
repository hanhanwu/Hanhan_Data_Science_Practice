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
* [Best tutorial][7]
### [My practice code][1]
1. Choose metrics, more invariant metric more likely to get significant results
2. Confirm α (significant leve, false positive rate)， β (1-power, false negative rate), min detectable effect (the lift experiment group got from control group), baseline perforamnce of selected metrics for control group
  * `Absolute min detectable effect = experiment group performance - control group performance`
  * `Relative min detectable effect = (experiment group performance - control group performance) / control group performance`
3. Estimation min sample size, applies for both groups
4. Estimate duration to collect enough data samples, making sure the data is using the same model params
5. Analysis, check whether there's significant difference (statistically significant, practically significant), also check whether there's significant lift
* [More tips][9]
* [More suggestions to comapre the control and the test groups][8]
* If your metrics is conversion rate, this tool gets min sample size easily: https://www.evanmiller.org/ab-testing/sample-size.html
  * [How to use this tool][10] 


[1]: https://github.com/hanhanwu/Hanhan_Data_Science_Practice/blob/master/Applied_Statistics/ABTest_Experiments/detailed_ABTest_Experiment.ipynb
[2]:https://github.com/dwyl/learn-ab-and-multivariate-testing
[3]:https://www.smashingmagazine.com/2010/06/the-ultimate-guide-to-a-b-testing/
[4]:https://www.smashingmagazine.com/2011/04/multivariate-testing-101-a-scientific-method-of-optimizing-design/
[5]:https://github.com/hanhanwu/Hanhan_Data_Science_Practice/blob/master/Applied_Statistics/Learning_Notes/AB_Test.md
[6]:https://github.com/hanhanwu/Hanhan_Data_Science_Practice/blob/master/Applied_Statistics/Learning_Notes/bayesian_analysis/Bayesian_Approaches_for%20Hypothesis_Tests.md
[7]:https://www.kaggle.com/code/tammyrotem/ab-tests-with-python/notebook
[8]:https://www.analyticsvidhya.com/blog/2021/03/a-b-testing-measurement-frameworks%e2%80%8a-%e2%80%8aevery-data-scientist-should-know/?utm_source=feedburner&utm_medium=email&utm_campaign=Feed%3A+AnalyticsVidhya+%28Analytics+Vidhya%29
[9]:https://towardsdatascience.com/simple-and-complet-guide-to-a-b-testing-c34154d0ce5a
[10]:https://guessthetest.com/calculating-sample-size-in-a-b-testing-everything-you-need-to-know/
