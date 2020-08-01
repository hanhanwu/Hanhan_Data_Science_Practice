# ANOVA 🌺


## ANOVA vs t-test
* t-test - It compares the means of a condition between 2 groups.
* ANOVA - It compares the means of a condition between 2+ groups.
* H0 & H1
  * H0: There is no significant difference among the groups.
  * H1: The difference between groups is significant.
* t-test and Post-ANOVA test
  * Post-ANOVA test is to find which groups are significantly different from each other
  * It uses t-tests to examine the mean differences between groups
  
## About Null Hypothesis
* F-ratio
  * It's the output of ANOVA, it allows to determine the variability "between samples" and "within samples" for multiple groups.
  * When the p-value associated with the F-ratio < significance level, reject null hypothesis.
    * [p-value calculator from F-ratio][2]
<p align="left">
<img src="https://github.com/hanhanwu/Hanhan_Data_Science_Practice/blob/master/Applied_Statistics/images/one_way_ANOVA_fscore.png" width="700" height="600" />
</p>

## 3 Types of ANOVA Test
* One Way ANOVA - It has 1 independent variable
* Two Way ANOVA - It has 2 independent variables
  * It examines the interaction between the two independent variables.
  * "Interactions" indicate that differences are not uniform across all categories of the independent variables.
  * [More description about With/Without Replication][5]
* N-Way ANOVA - It has 2+ independent variables


## Assumptions
1. The observations are obtained independently and randomly from the population, by factor levels
  * "factor level" indicates different categoryies of a variable, same as the factor variable in R. For example you can divide age into different factor levels such as child, teenager, adult and seniors.
2. Normality, the data for each factor level is normally distributed
  * Can be validated through histograms, skewness & kurtosis, Q-Q plot, [Shapiro-Wilk][3] or KS score
3. Sample cases are independent from each other
4. Homogeneity, the variance of each group are almost the same
  * Can be validated through [Levene test][4] or Brown-Forsythe Test (using `median` in python Levene test)
* The violation of assumptions
  * Violate the homogeneity, you can still trust the results but make sure each group has the equal size
  * Violate the Cases Independence, the result will be invalid
  * Violate the Normality, you can still trust the results if the sample size is large


## Reference
* [ANOVA Statistics][1]
  * There are some mistakes in this tutorial that're misleading... that's why I'm using another dataset with a different example.

[1]:https://www.analyticsvidhya.com/blog/2020/06/introduction-anova-statistics-data-science-covid-python/?utm_source=feedburner&utm_medium=email&utm_campaign=Feed%3A+AnalyticsVidhya+%28Analytics+Vidhya%29
[2]:https://www.socscistatistics.com/pvalues/fdistribution.aspx
[3]:https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.shapiro.html
[4]:https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.levene.html
[5]:http://www.biostathandbook.com/twowayanova.html#:~:text=A%20two%2Dway%20anova%20is,one%20female%20of%20each%20genotype.
