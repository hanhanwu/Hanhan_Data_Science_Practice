# p-value vs t-test vs z-test

## A Good Summmary
<p align="center">
<img src="https://github.com/hanhanwu/Hanhan_Data_Science_Practice/blob/master/Applied_Statistics/images/p_value.PNG" width="700" height="400" />
</p>

* When `p-value < α`, it's within the critical region, reject null hypothesis (H0)
* Or when `test statistical > critical value`, reject null hypothesis (H0)
  * test statistical means z-score or t-score
  * Critical value maps to α: https://www.stat.colostate.edu/inmem/gumina/st201/pdf/Utts-Heckard_t-Table.pdf
    * We often choose α in [0.01, 0.05, 0.1] for 2 tailed tests
      * such as α=0.05 and the critical value is 1.96
* z-test vs t-test
  * The difference is just in polulation variance vs sample variance
  * When population variance is unknown -> Use t-test
    * We can use sample variance
  * When population variance is known -> z-test
    
## z-test
### One Sample z-test
* The hypothesis tests compares a sample means with an assumped mean (population mean)
<p align="center">
<img src="https://github.com/hanhanwu/Hanhan_Data_Science_Practice/blob/master/Applied_Statistics/images/z_test.PNG" width="700" height="400" />
</p>
* NOTE
  * When we are using z_score just to compare a single value xi and the expected value of X, we use `z = (xi - mean)/std`
    *  if z > 0, larger z means xi is less likely to be smaller than the expected value
    *  if z < 0, larger abs(z) means xi is more likely to larger than the expected value

### Two Sample z-test
* The hypothesis tests compares the sample mean difference between 2 samples with an assumped mean different (population mean difference)
<p align="center">
<img src="https://github.com/hanhanwu/Hanhan_Data_Science_Practice/blob/master/Applied_Statistics/images/z_test_2sample.PNG" width="700" height="400" />
</p>


## t-test
### One Sample t-test
* The hypothesis tests compares a sample means with an assumped mean (population mean)
<p align="center">
<img src="https://github.com/hanhanwu/Hanhan_Data_Science_Practice/blob/master/Applied_Statistics/images/t_test.PNG" width="700" height="400" />
</p>

### Two Sample t-test
* The hypothesis tests compares the sample mean difference between 2 samples with an assumped mean different (population mean difference)
<p align="center">
<img src="https://github.com/hanhanwu/Hanhan_Data_Science_Practice/blob/master/Applied_Statistics/images/t_test_2sample.PNG" width="700" height="400" />
</p>


## Reference
* https://www.analyticsvidhya.com/blog/2020/06/statistics-analytics-hypothesis-testing-z-test-t-test/?utm_source=feedburner&utm_medium=email&utm_campaign=Feed%3A+AnalyticsVidhya+%28Analytics+Vidhya%29
  * Best reference ever
