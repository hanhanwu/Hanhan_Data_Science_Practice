# Feaure Selection Collection
In industry, many times we need to generate features, understanding them and generate more to improve model performance. I'm taking notes of some method that may help do further exploration.

## SHAP
* It's amethod used to deal with the draw back of XGBoost feature selection
* [You know what, SHAP value came from game theory][2]
  * "Shapley values correspond to the contribution of each feature towards pushing the prediction away from the expected value."
* [My practice code 1 - XGBoost Regressor][1]
* Github has blocked the loading of JS, in fact it provides a method to interact with each record and understand how the feature values affect the prediction
![shap JS](https://github.com/hanhanwu/Hanhan_Data_Science_Practice/blob/master/Better4Industry/Feature_Selection_Collection/xgboost_shap.PNG)

  * In this force plot, the "base value" of the average predicted value from the model training, the "output value" is the predicted value of current observation. Pink is the position impact while blue is the negative impact. Both impacts indicate how does the base value distance from the output value
* [SHAP decision plot][3]
  * [Code example][4]
    * Just need trained model and feature values, no need lables in the data
    * Add `link='logit'` in the `decision_plot()` function will convert log odds back to prediction probabilities (SHAP value is log odds)
  * [SHAP decision source code][5]
    * In the `decision plot`, by default the features are ordered by importance order. When there are multiple records, the importance of a feature is the sum of absolute shap values of the feature
  * Include multiple records plot
  * For each record, it shows how does each feature value lead to the final predictio result

## Tips
* When the data is large, you can use `clustered_df = shap_kmeans(df)` and put this clustered_df in a shap explainer. This method helps speed up the computation
  * In SHAP, for each feature subset (2^m - 2) it perturbs the values of features and makes prediction to see how peturbing a feature subset changes the prediction of model. For each feature subset (e.g. [0,1,1,0,0,0] only perturbing feature 2nd and 3rd) you can replace the feature values by any of the values in the training set. By default it does that exhaustively for all points in training, therefore the total number of model predictions it evaluates is n2^m. <b>So, we use shap.kmeans to only perturb based on some representatives (10 centroids instead of 1000 datapoints)</b>
    * `m` is the feature number
    * `N` is the number of samples
    * Total number of model evaluation is `N * (2^m - 2)` 
    * Although the value is randomly assigned for the perturbed values, it's choosing the possible value from the feature that appeared in the training data 
  * There are [different types of SHAP explainer][6]
    * `KernelExplainer` is generic and can be used for all types of models, but slow. 
      * That's also why when using TreeExplainer, you don't have to use `shap.kmeans` for large dataset, since it's fast 
      * KernelExplainer is not applicable for more than 15 features
* When doing the experiments of SHAP performance, there are multiple things can check 
  * Time efficiency for different number of samples, differnt number of features, different model sizes (such as different tree numbers)
  * While the time efficiency has been improved, how's the accuracy of model predictions 
* I don't think SamplingExplainer can replace TreeExplainer for ensembling models
  * It requires a model input from `train()`, and cannot use loaded trained model
  * When there are 3000+ samples, TreeExplainer is still faster  
* Display shap plots on Databricks Notebooks

```
import matplotlib.pyplot as plt

exp_shap = shap.TreeExplainer(model)
shap_tree = exp_shap.shap_values(X_test_baseline)
expected_tree = exp_shap.expected_value
if isinstance(expected_tree, list):
  expected_tree = expected_tree[1]
print(f"Explainer expected value: {expected_tree}")

idx = 10

print(f'Tree Explanations for #"{idx}" observation in test dataframe:')
## Option 1
shap_force_plot = shap.force_plot(expected_tree, shap_tree[idx], feature_names=baseline_X_cols, matplotlib=True) # This doesn't show feature values
## Option 2
shap_force_plot = shap.force_plot(tree_explainer.expected_value, shap_values[check_row,:], X.iloc[check_row,:], matplotlib=True)  # this will show feature values, but look messy
display(shap_force_plot)
```


[1]:https://github.com/hanhanwu/Hanhan_Data_Science_Practice/blob/master/Better4Industry/Feature_Selection_Collection/try_shap_xgboost.ipynb
[2]:https://www.analyticsvidhya.com/blog/2019/11/shapley-value-machine-learning-interpretability-game-theory/?utm_source=feedburner&utm_medium=email&utm_campaign=Feed%3A+AnalyticsVidhya+%28Analytics+Vidhya%29
[3]:https://towardsdatascience.com/introducing-shap-decision-plots-52ed3b4a1cba
[4]:https://slundberg.github.io/shap/notebooks/plots/decision_plot.html
[5]:https://github.com/slundberg/shap/blob/6af9e1008702fb0fab939bf2154bbf93dfe84a16/shap/plots/_decision.py#L46
[6]:https://shap-lrjball.readthedocs.io/en/docs_update/api.html
