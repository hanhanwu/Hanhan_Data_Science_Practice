Although everything I created in my GitHub are good for industry used, today I decided to create a folder to put those technologies, tools, methods, etc. that can better serve for Industry work.

****************************************************************************************

### ACCURACY & INTERPRETABILITY

* Lime - Visualize feature importance for all machine learning models
  * Their GitHub, Examples and the Paper: https://github.com/marcotcr/lime
  * The tool can be used for both classification and regression. The reason I put it here is because it can show feature importance even for blackbox models. In industry, the interpretability can always finally influence whether you can apply the more complex methods that can bring higher accuracy. Too many situations that finally the intustry went with the most simple models or even just intuitive math models. This tool may help better intrepretation for those better models.
  * My code: https://github.com/hanhanwu/Hanhan_Data_Science_Practice/blob/master/Better4Industry/lime_interpretable_ML.ipynb
    * It seems that GitHub cannot show those visualization I have created in IPython. But you can check LIME GitHub Examples
    * LIME requires data input to be numpy array, it doesn't support pandas dataframe yet. So that's why you can see in my code, I was converting the dataframe, lists all to numpy arraies.
