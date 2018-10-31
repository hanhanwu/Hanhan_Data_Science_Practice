## Experience Notes

### Machine Learning Workflow Related
#### Preprocessing
* When removing those features with low variance, threshold better not to be very high. Some features have lower variance but plays more important role in the data
#### Bootstrap the whole pipeline
* Even after building the whole machine learning pipeline, you need to run it multiple times with different settings (such as different seeds), and calculate the average & median & variance of multiple rounds final results, to avoid you got very higher or very low result by accident.
#### About Data Spliting
##### Hold-out data
* We know that we use cross validation to reduce overfitting. Meanwhile we can use Hold-out data to keep reducing it.
* In your data, maybe you can split it into Train, Test and Validation data. With Train-test you can use cross valiation and other methods, do multiple rounds of improvement. Validation data is always out side there, and you evaluate on both train-test and validation to see whether they get similar evaluation results.

#### Reset Index
* Pay attention to those python methods which will shuffle the index in the original data, such as train-test-spliting, especially after you used stratified or shuffle.
* To reset index, use `df.reset_index(drop=True, inplace=True)`, after resetting, the index of rows will start from 0

#### About Sampling
* In python, we majorly use `imblearn` to do sampling. No matter it's oversampling or undersampling, it can have synthetic records generated.
* The so called right practice is, you should do train-test spliting before sampling, and only apply sampling on training data, and leave testing data having all original records. This will bring a problem, especially when your data is severely imbalanced.
  * Most of imblearn methods will generate balanced dataset by default. In fact, even if you try to set `ratio` by specifying which class occupies how much ratio, imblearn tends to keep giving you errors, very tricky (open source...)
  * With balanced training data, it's not bad, at least it can reduce the influence of the majority class in order to reduce bias. But the problem is. your testing data is all original data, it's still imbalanced.
    * A commonly used practice is stratified train-test spling, so that both training and testing data will have similar distrubution of each class. Therefore, your testing data is still imbalanced.
  * While training on the sampled training data, it's also worthy to do some verification:
    * Track original and synthetic records from the sampled training data first, you can create a new label, called "is_original".
    * Then use stratified k fold spliting to split the data into k folds, better to split by both original label and "is_original" label. In each fold, check model preformance for original data and synthetic data, finally get some aggregated performance metrics for all folds.
      * The reason that better to split with "is_original" label here is because, you are trying to check both original data performance and synthetic data performance, without split by this label, there can be folds that do not have original/synthetic data at all, which will influence the performance report.
  * Of course, validating sampled training data won't really help you improve testing preformance, it just provides some insights. If your testing data gets very bad performance results, here are something you can try:
    * Heavily rely on feature engineering, try to find the right features.
    * Use "is_original" as a feature.
    * Both training and testing data uses multi-label, which is the combination of original label and "is_original". But when calculating testing performance, ignore "is_original" label.
      * In python, some sklearn methods does not support multi-label format, such as `StratifiedKFold`, you need to convert multi-label into 1 dimensional string array
    * Anomalous Detection, especially when the data is severely imbalanced.
    * If you got very low precision or recall, which maybe caused by oversampled minority class, maybe try undersampling, even just to randomly select a subset and run the whole pipeline multiple rounds
  * When you are using oversampling, better to check whether the synthetic records are all added behind the original data or they are inserted among original data. If it's inserted in between, and difficult to find any way to join the sample data with original data (in order to get is_original label), then it can be troublesome
