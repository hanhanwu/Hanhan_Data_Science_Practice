# Chi-Square Test 🔮

💖For categorical variables. Code can be found [here][3]. 💖

## About Chi-Square Test
* Chi-Square test is a test of statistical significance for Categorical Variables.
  * Categorical Variables
    * Nominal variable - no natural ordering of categorical values
    * Ordinal variable - categories can be placed in order
* Chi-Square test in hypothesis testing is used to test the hypothesis about the distribution of observations/frequencies in different categories.
* <b>Assumptions of Chi-Square Test</b>
  * Data is randomly picked from the population
  * Categories are mutually exclusive
    * Each observation only belongs to 1 category
  * Observations are also independent from each other
  * The data is in frequency or count format for each category, but should not be percentage format
  * If more than 20% frequencies have less than 5 observations, chi-square should not be used
    * Either combine categories
    * Or get more data

## How to Calculate Chi-Square Value
* Step 1 - Null Hypothesis
  * Null Hypothesis: The observed frequencies are NOT significant different from the expected frequencies
  * Alternative Hypothesis: The observed frequencies are significant different from the expected frequencies
* Step 2 - Chi-square value `χ2 = sum(power(Oi - Ei, 2)/Ei)`
  * Oi - Observed frequency
  * Ei - Expected frequency
* Step 3 - Check [Chi-Square Table][1]
  * Degrees of freedom (df) = number of categories - 1
  * Based on df and α (such as 5% significant level), find the critical value in the table
  * If chi-square larger than the critical value, reject Null Hypothesis


## Pearson’s Chi-Square Test for Association

### About
* It checks the association/independence between 2 categorical variables

### How to Calculate
* Step 1 - Null Hypothesis
  * Null Hypothesis: The 2 variables are independent
  * Alternative Hypothesis: The 2 variables are dependent
* Step 2 - Expected Freqnecy
  * `Ei = (Row Total x Column Total)/Grand Total`
* Step 3 - Chi-square value `χ2 = sum(power(Oi - Ei, 2)/Ei)`
  * Oi - Observed frequency
  * Ei - Expected frequency
* Step 4 - Check [Chi-Square Table][1]
  * Degrees of freedom (df) = (total rows - 1) * (total columns - 1)
  * Based on df and α (such as 5% significant level), find the critical value in the table
  * If chi-square larger than the critical value, reject Null Hypothesis

### [Reference][2]


[1]:https://people.smp.uq.edu.au/YoniNazarathy/stat_models_B_course_spring_07/distributions/chisqtab.pdf
[2]:https://www.analyticsvidhya.com/blog/2019/11/what-is-chi-square-test-how-it-works/?utm_source=feedburner&utm_medium=email&utm_campaign=Feed%3A+AnalyticsVidhya+%28Analytics+Vidhya%29
[3]:https://github.com/hanhanwu/Hanhan_Data_Science_Practice/blob/master/Applied_Statistics/Learning_Notes/chi_square_test.R
