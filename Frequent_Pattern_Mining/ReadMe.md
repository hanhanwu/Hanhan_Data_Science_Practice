
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
* Classification based on Frequent Pattern Mining
  * Frequent patterns map data to a higher dimensional space. They capture more underlying semantics of the
data, and thus can hold greater expressive power than single features.
  * Through frequent pattern based classification, we are actually transformed the original feature space to a larger space, which may include the chance of including important features.
  * Many of the frequent patterns generated in frequent itemset mining are indiscrimina- tive because they are based solely on support, without considering predictive power.
  * The general framework for discriminative frequent pattern–based classification
    * Feature Generation - partition the data according to the class label, discover frequent patterns that satisfy min support in each partition. The collected frequent patterns are feature candidates
    * Feature Selection - Information gain, Fisher score, or other evaluation measures can be used for this step. Can also have relevancy checking to weed out redundant patterns
    * Classification - Just apply classifiers
  * DDPMine (direct discriminative pattern mining) directly mines the discriminative patterns and integrates feature selection into the mining framework. The theoretical upper bound on infor- mation gain is used to facilitate a branch-and-bound search, which prunes the search space significantly. Experimental results show that DDPMine achieves orders of mag- nitude speedup over the two-step approach without decline in classification accuracy.
    * I cannot find any built-in library


*************************************************************

ALGORITHMS REVIEW

* Apriori
  * It is almost the most basic alg in frequent mining, but it simply works. If, performance is important, Apriori may be slower than some other alg
  * [R] code: https://github.com/hanhanwu/Hanhan_Data_Science_Practice/blob/master/Frequent_Pattern_Mining/basic_alg_apriori.R
    * This R package could recommend you what items to put together, and show you support, confidence and lift at the same time
  * reference: https://www.analyticsvidhya.com/blog/2017/08/mining-frequent-items-using-apriori-algorithm/?utm_source=feedburner&utm_medium=email&utm_campaign=Feed%3A+AnalyticsVidhya+%28Analytics+Vidhya%29
  
  
* FP-Growth
  * [Spark Python] code: https://github.com/hanhanwu/Hanhan_Data_Science_Practice/blob/master/Frequent_Pattern_Mining/association_rules.ipynb
   * Looks like the most esay-to-use python library for association rules, although how to use that `where` clause in this case casued my some time...
     * https://stackoverflow.com/questions/50559308/spark-python-cannot-match-array

*************************************************************

SPMF

* SPMF is a Java library, it contains many frequent mining algorithms that you  won't find in either python nor R, but SPMF really has so many advanced algorithms. So why not just use Java.
* Download, Install & Run
  * First of all, you need to download and install Eclipse: http://www.eclipse.org/downloads/packages/release/
    * Download the latest released version (I think Eclipse is very good at giving cool names to its release), and install Java IDE or J2EE
    * Download and install Java JDK: http://www.oracle.com/technetwork/java/javase/downloads/jdk8-downloads-2133151.html
    * Then when you have launched Eclipse, create a project name/path, click "Next" and choose "Use a project specific JRE" and select the JDK version you installed. It may also better NOT to create your workspace in a folder that may have more access restriction. 
  * Install source code version: http://www.philippe-fournier-viger.com/spmf/index.php?link=download.php
    * You can follow the steps here: http://www.philippe-fournier-viger.com/spmf/how_to_install.txt
    * In my case, after upcompress the zip folder, you need to move folder `ca` and the license directly under `src`
    * After clicking `Refresh` of `src`, you may still get many warnings. But just go to `ca.pfv.spmf.tests` folder, find a test file such as `MainTestCharm_saveToMemory.java`, run as application. If the execution succeeded, you should be fine.
  * My Practice code [Java]: https://github.com/hanhanwu/Hanhan_Data_Science_Practice/blob/master/sequencial_analysis/TryCPTPlus.java
    * training data: https://github.com/hanhanwu/Hanhan_Data_Science_Practice/blob/master/sequencial_analysis/training.txt
    * HOW to run the code:
      * Check how to install `SPMF` here: https://github.com/hanhanwu/Hanhan_Data_Science_Practice/tree/master/Frequent_Pattern_Mining
      * create a new folder under `ca.pfv.spmf.test` called "hanhan_test". You can use other folder name, but need to change the path in the .java file above.
      * put both training data and the code into "hanhan_test" folder
      * run the .java file as application
    * Description of `TryCPTPlus.java`
      * The original implementation came from https://github.com/tedgueniche/IPredict
      * Open source is open source.... it has very strict requirements for the input data format:
        * elements in each sequence have to be numbers
        * Each number has to be seperated by " -1 ", and the end of the sequence should be " -1 -2"
          * This will create the limit that your numbers cannot be -1 or -2
      * The testing data is imput by yourself in line 56, 57, 58. In my code, as you can see I input "2,4" to predict which number should follow them
      * One of the bug of this open source code is, if the prediction has more then 1 element that got the same score, the prediction returns empty value....
