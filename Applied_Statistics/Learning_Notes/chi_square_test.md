# Chi-Square Test 🔮

💖For categorical variables.💖

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
* Chi-square value `χ2 = sum(power(Oi - Ei, 2), Ei)`
  * Oi - Observed frequency
  * Ei - Expected frequency
