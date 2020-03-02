# A/B Tets

🌺 The main purpose for A/B test is to use the data we collect from variants A and B, computing some metrics for each variant, then with statistical methods to determine which variant is better.

## Frequentist Methods
* [My Code - Practical A/B Test Experiments][1]
* Using p-value to choose hypothesis
  * Null Hypothesis - There's no difference between variant A, B
  * Alternative Hypothesis - A, B are different
* "A p-value measures the probability of observing a difference between the two variants at least as extreme as what we actually observed, given that there is no difference between the variants."
  * Only after p-value achieves statistical significance or we have seen enough data, the experiment ends.
* <b>Drawbacks</b>
  * Have to know how much data needed.
  * It's hard to know what to do when it's not statistically significant, or you don't have enough data.
  * Cannot test in real time, but need to wait till getting planned data size.
  * Not sure what to do when the new solution is slightly better than the old solution.

## Bayesian A/B Test
* In Bayesian A/B testing, we model the metric for each variant as a random variable with some probability distribution. 
* After observing data from both variants, we update our prior beliefs about the most likely values for each variant.
* Comparing with Frequentist Method, Bayesian method tend to accept variants that offer slight improvement. Bayesian A/B testing asserts that the false positive rate (the proportion of times we accept the solution when the it's not really better) is not very important. Instead, Bayesian A/B testing focuses on the average magnitude of wrong decisions over the course of many experiments.
* In a word, Bayesian A/B test tends to guaratee the long term improvement on a metric, by limiting the amount of wrong decisions that can make the products worse.
* For industries that did lots of experiments, they might have enough priori distributions to use, this could help reach to the conclusion faster with less data.
  * But when choosing the priori, better to choose a weaker distribution then the historical observations suggested.
  * Bayesian A/B test reaches to the conclusion faster than other methods
* Drawbacks:
  * Difficult to explain the concept of expected loss to business audience.

* References
  * https://medium.com/convoy-tech/the-power-of-bayesian-a-b-testing-f859d2219d5
  * https://blog.exploratory.io/an-introduction-to-bayesian-a-b-testing-in-exploratory-cb5a7ad80963
* R - `bayesAB`
  * R package: https://github.com/FrankPortman/bayesAB


[1]:https://github.com/hanhanwu/Hanhan_Data_Science_Practice/tree/master/Applied_Statistics/ABTest_Experiments
