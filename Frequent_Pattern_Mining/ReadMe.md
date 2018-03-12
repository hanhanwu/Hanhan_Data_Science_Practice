
Experiments here are going to review those basic/popular frequent mining algorithms and also to explore new and better algorithms

*************************************************************

CONCEPTS & TERMINOLOGY

* Frequent Pattern Mining is an unsupervised learning
* <b>Support</b> - probability of an event to occur. For example, `Support(ice-cream) = number of transactions that contain ice-cream/totoal number of transactions`
* <b>Confidence</b> - conditional probability. For example, the probability that you will buy ice-cream after you have bought chocolate
* <b>Lift</b> - The ratio of confidence to expected confidence. <b>The probability of all of the items in a rule occurring together (otherwise known as the support) divided by the product of the probabilities of the items on the left and right side occurring as if there was no association between them.</b>
* Association Rules 2 steps
  * Frequent itemset mining
  * Rule Generation
    * All association rules must satisfy certain condistions (confidence), and the proportion of the dataset that they actually represent


*************************************************************

ALGORITHMS REVIEW

* Apriori
  * It is almost the most basic alg in frequent mining, but it simply works. If, performance is important, Apriori may be slower than some other alg
  * [R] code: https://github.com/hanhanwu/Hanhan_Data_Science_Practice/blob/master/Frequent_Pattern_Mining/basic_alg_apriori.R
    * This R package could recommend you what items to put together, and show you support, confidence and lift at the same time
  * reference: https://www.analyticsvidhya.com/blog/2017/08/mining-frequent-items-using-apriori-algorithm/?utm_source=feedburner&utm_medium=email&utm_campaign=Feed%3A+AnalyticsVidhya+%28Analytics+Vidhya%29
