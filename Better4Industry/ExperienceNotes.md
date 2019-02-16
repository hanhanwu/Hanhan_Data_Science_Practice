# Experience Notes

## Avoid Mistakes Notes
* Validate source data quality
* Be careful for choosing part of the labels
  * Example - You have "fraud" and "non-fraud" as the labels, meanwhile the client gave you "approve" and "decline" to tell you human decision before knowing which is the fraud. To simplify the problem, you may just choose "fraud" and "approved nonfraud", therefore you dind't include "declined nonfraud". Next when you are generating the features, some features may include all the declined cases and in your prediction, all those declined fraud will be predicted as fraud, looks like good performance but in fact included the declinded nonfraud, however since it's not in your labels, you are <b>overfitting without noticing from evaluation metrics</b>.
  * If you are not sure which labels to include, there is a good practice that is to predict everything with probility, then compare the distribution for different labels. This will give you more insights about which labels should be used in the prediction.

## Cross Clients Notes
* Same model, Similar case
  * About Feature Importance - Difference clients may have difference feature importance, even your features are generated in the same way, the cases are similar and you will use the same model (can be different param, though). So better NOT to use one client's important features to serve as another client's important features. If there is time, if you will get labeled data, seperate the case for each client.
  * Impact of the feature value - Even we have decied to used the same batch of features to serve for different clients, we can check the value range of each feature that impacts the prediction, to see whether it's the same for each feature. For example, feature F1, its higher value has more impact on Class 0 for client A, but its lower value has more impact on Class 0 for Client B. The difference may make the same model, same feature set won't work cross clients.

## Machine Learning Workflow Related
### Data Collection
* When generating aggregated features, you can try not only avg, sum, median, std, etc., but also central tendency related such as values within [mean-std, mean+std], [mean-2std, mean+2std], [mean-3std, mean+3std]; you can also try percentile, such as only collect first 25%, last 25%, etc.
  * But these will also create large amount of highly correlated features. Simply remove highly correlated features might instead remove the more important one. A method we can try is, to use tree models to generate feature importance (such as SHAP) with all the features, then calculate the correlation and throw away the nonimportant ones.
  * Also be careful, for highly correlated features, if one is important, the other can also rank high in tree models

### Preprocessing
#### Normalization
* [How outliers influence normalization methods][1]
  * The way it measures each scaler is to check whether after normalization, the data distribution is more balanced/less skewed/more gaussian-like
#### Remove Low Variance Features
* Better to normalize the data before checking variance, larger values tend to have larger variance
* When removing those features with low variance, threshold better not to be very high. Some features have lower variance but plays more important role in the data
### Bootstrap the whole pipeline
* Even after building the whole machine learning pipeline, you need to run it multiple times with different settings (such as different seeds), and calculate the average & median & variance of multiple rounds final results, to avoid you got very higher or very low result by accident.
### About Data Spliting
#### Hold-out data
* We know that we use cross validation to reduce overfitting. Meanwhile we can use Hold-out data to keep reducing it.
* In your data, maybe you can split it into Train, Test and Validation data. With Train-test you can use cross valiation and other methods, do multiple rounds of improvement. Validation data is always out side there, and you evaluate on both train-test and validation to see whether they get similar evaluation results.

### Reset Index
* Pay attention to those python methods which will shuffle the index in the original data, such as train-test-spliting, especially after you used stratified or shuffle.
* To reset index, use `df.reset_index(drop=True, inplace=True)`, after resetting, the index of rows will start from 0

### About Sampling
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
  
### About Prediction
* When you are predicting the probability, you can adjust the estimator threshold. Sklearn default threshold is 0.5, but if the dataset is not balanced, then 0.5 is not the case.
  * <b>But if you can checking any results before prediction results, such as feature importance using `fit()`, this method won't help</b>. I met a situation that with this method, you may get better prediction results, but in fact you training results may have lower precision or recall.
* When using sampling, built-in sklearn methods may not work well in many data imbalancing cases. We can also try to set class_weight in the estimator. Setting it as balanced may not work well, sometimes you may need to set the majority class with lower ratio

## Model Cheatsheet
### XGBoost
* Python XGBoost has different methods to save model
  * `save_model()`, `save_binary()`, `pickle.dump(your_model)`, `dump_model()` will return the artifact
  * NOTE! - If you want artifact, you have to use python xgboost, the one NOT sklearn API
* After you saved the model, normally in other languages, there is `loadModel()` method to call the trained model directly.
  * Such as XGBoost php wrapper: https://github.com/bpachev/xgboost-php
* Production team may try to understand how XGBoost work
  * https://docs.aws.amazon.com/sagemaker/latest/dg/xgboost-HowItWorks.html
    * Its sublinks at the bottom are also pretty good
  * How Gradient descent work: https://towardsdatascience.com/gradient-descent-in-a-nutshell-eaf8c18212f0
* How to convert artifact to prediction result
  * The artifact in XGBoost is a text file, which shows each tree in XGBoost. The leaf score is what you want, but need some convertion.
  * The leaf score the the objective socre, see `objective` here: https://xgboost.readthedocs.io/en/latest/parameter.html
    * You need to sum up leaf values first
    * Then based on the objective, convert the total leaf value to prediction probability
      * For example, you are using "binary: logistic" as your objective, use the conversion rule here to convert logit score to probability: https://sebastiansauer.github.io/convert_logit2prob/
* How to manually tune XGBoost Param (use it when time is limited): "initialize parameters such: eta = 0.1, depth= 10, subsample=1.0, minchildweight = 5, colsamplebytree = 0.2 (depends on feature size), set proper objective for the problem (reg:linear, reg:logistic or count:poisson for regression, binary:logistic or rank:pairwise for classification)split %20 for validation, and prepare a watchlist for train and test set, set num_round too high such as 1000000 so you can see the valid prediction for any round value, if at some point test prediction error rises you can terminate the program running,i) play to tune depth parameter, generally depth parameter is invariant to other parameters, i start from 10 after watching best error rate for initial parameters then i can compare the result for different parameters, change it 8, if error is higher then you can try 12 next time, if for 12 error is lower than 10 , so you can try 15 next time, if error is lower for 8 you would try 5 and so on.ii) after finding best depth parameter, i tune for subsample parameter, i started from 1.0 then change it to 0.8 if error is higher then try 0.9 if still error is higher then i use 1.0, and so on.iii) in this step i tune for min child_weight, same approach above,iv) then i tune for colSamplebytreev) now i descrease the eta to 0.05, and leave program running then get the optimum num_round (where error rate start to increase in watchlist progress)" - from a Kaggle master
  
## Production Handover Notes
* Understand deployment QA at the very beginning. Provide every output at each step for their QA test.
* Push everyone in the team to put all the code together on time, after combining the code, re-run everything asap, to reduce potential within-team discrepencies.
  
[1]:http://scikit-learn.org/stable/auto_examples/preprocessing/plot_all_scaling.html
