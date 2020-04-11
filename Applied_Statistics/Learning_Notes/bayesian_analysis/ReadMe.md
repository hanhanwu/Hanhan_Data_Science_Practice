# Bayesian Analysis

More details about Bayesian Analysis


## Reference
* [Doing Bayesian Data Analysis][2] - It's a book

## Bayesian Theorem vs Conditional Probability
* Bayesian Theorem
  * `P(A|B) = P(B|A)*P(A)/P(B)`
* Conditional Probability
  * `P(A|B) = P(A and B)/P(B)`
* Both can be used to calculate probability, and both have independence assumption is there are many subcases for A or B.


## Bayesian Approach for Hypothesis Analysis
* [Concepts][4]
* [How to use Exploratory (a tool) to do Bayesian A/B test with/without prior][5]
  * [Exploratory][6]

## R Analysis
* [Running Proportion][1]
  * It's flip coin problem, the sequence of binary values. `Running Proportion = cumulated sum / cumulated count`, with running proportion, it can help display the trending of a coin's head (or tail) frequency when the numer of flips is increasing.
  
* [Integral of Density][3]
  * The integral of this normal distribution is the approximation of the sum of width * height from each interval


[1]:https://github.com/hanhanwu/Hanhan_Data_Science_Practice/blob/master/Applied_Statistics/Learning_Notes/bayesian_analysis/bayesian_running_proportion.R
[2]:https://www.amazon.com/Doing-Bayesian-Data-Analysis-Tutorial/dp/0123814855/ref=cm_cr_arp_d_product_top?ie=UTF8
[3]:https://github.com/hanhanwu/Hanhan_Data_Science_Practice/blob/master/Applied_Statistics/Learning_Notes/bayesian_analysis/bayesian_integralOfdensity.R
[4]:https://github.com/hanhanwu/Hanhan_Data_Science_Practice/blob/master/Applied_Statistics/Learning_Notes/bayesian_analysis/Bayesian_Approaches_for%20Hypothesis_Tests.md
[5]:https://blog.exploratory.io/an-introduction-to-bayesian-a-b-testing-in-exploratory-cb5a7ad80963
[6]:https://exploratory.io/?utm_campaign=ab_test&utm_medium=blog&utm_source=medium
