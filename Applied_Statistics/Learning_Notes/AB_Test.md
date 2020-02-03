# A/B Tets

🌺 The main purpose for A/B test is to use the data we collect from variants A and B, computing some metrics for each variant, then with statistical methods to determine which variant is better.

## Frequentist Methods
* Using p-value to choose hypothesis
  * Null Hypothesis - There's no difference between variant A, B
  * Alternative Hypothesis - A, B are different
* "A p-value measures the probability of observing a difference between the two variants at least as extreme as what we actually observed, given that there is no difference between the variants."
  * Only after p-value achieves statistical significance or we have seen enough data, the experiment ends.
* <b>Drawbacks</b>
  * Have to know how much data needed.
  * It's hard to know what to do when it's not statistically significant, or you don't have enough data.
  * Cannot test in real time, but need to wait till getting planned data size.
