Although everything I created in my GitHub are good for industry used, today I decided to create a folder to put those technologies, tools, methods, etc. that can better serve for Industry work.

## [Experience Notes][1]

## Building Industry Data Science Pipeline

### AWS Sagemaker
* With AWS Sagemaker, you can use multiple AWS services to store the data and running data science jobs in parallel, meanwhile Sagemaker provides notebook almost the same as IPython, and shared space for team development. It's just cost money, and you may need data engineers to help set up the whole pipeline.

### Luigi
* It's a free python library that allows you to build pipeline which will help control parallel running, and you can also schedule multiple same pipelines for different clients at the same time. It's convenient to run locally. For each of its step, storing the output data is required, but once the data is stored, this step will be skip later we you can running the code again. <b>Better to use luigi when you have confimed a pipeline and will use that repeatedly. If you begin with luigi for building the pipeline, can be less efficient.</b>
* Luigi Github: https://github.com/spotify/luigi
* Here's my sample code: https://github.com/hanhanwu/Hanhan_Data_Science_Practice/tree/master/Better4Industry/luigi_pipeline1
  * About this pipeline
    * For hard coded params, you can store them in .yaml config file
    * Data Input stores the original input data
    * Then in Data_Pred:
      * `task_read_raw.py` is just to tell where is the oringinal input data
      * `generate_base_data.py` is intended to generate a dataframe that can be used many times in the future, without re-reading the original data input. 
    * `Feature_Generation` will generate features from the output of `generate_base_data.py`
    * Finally, `execute.py` will just call 1 task, it will execute the whole pipeline
  * Each task has `requires()` to indicate which tasks have to be finished running before current task. `output()` is also required for each task, which will store the output in each task.
  * NOTES
    * Sometimes, not easy to debug
    * When you changed the code of a task, but if its output is stored there, Luigi will skip this task while running the pipeline and your output cannot be changed, which is inconvenient. You can write the code to remove the file automatically BEFORE running this task again. Like what I did in `execute.py`
    * It seems that location needs absolute path, as you can see in my code. I tried relative path, got errors, didn't work.
    * When a value in .yaml file is a very long string, it's fine to just use double quotes or single quotes for the string even if you want to use multiple lines to show the string. Don't use `"""..."""` in yaml. This is something different from python.
    * For a list `[...]` in .yaml file, python will read it as `(...)`, so in your python code, need to use `list()` to convert to a python list.
    * Sometimes the config file will use previous settings even after you change the key name, so need to check carefully
    * Pandas also could behave in a weird way, what I met was sometimes it will assign the value with `apply()` sometimes not, which could create more chanllenge in testing

## ACCURACY & INTERPRETABILITY

### Lime - Visualize feature importance for all machine learning models
* Their GitHub, Examples and the Paper: https://github.com/marcotcr/lime
* The tool can be used for both classification and regression. The reason I put it here is because it can show feature importance even for blackbox models. In industry, the interpretability can always finally influence whether you can apply the more complex methods that can bring higher accuracy. Too many situations that finally the intustry went with the most simple models or even just intuitive math models. This tool may help better intrepretation for those better models.
* My code: https://github.com/hanhanwu/Hanhan_Data_Science_Practice/blob/master/Better4Industry/lime_interpretable_ML.ipynb
  * It seems that GitHub cannot show those visualization I have created in IPython. But you can check LIME GitHub Examples
  * LIME requires data input to be numpy array, it doesn't support pandas dataframe yet. So that's why you can see in my code, I was converting the dataframe, lists all to numpy arraies.
* NOTES
  * Currently they have to use predicted probability in `explain_instance()` function
  * You also need to specify all the class names in `LimeTabularExplainer`, especially in classification problem, otherwise the visualization cannot show classes well
  
### SHAP
* It has more user friendly visualization than Lime. Lime indicates the feature value tends to be which class, in each record; SHAP provides an overall feature importance and also how each value affects the prediction.
* SHAP is also aiming at dealing with the draw back in XGBoost feature importance
* [More Details][2]
  
### Inspired by FastAI Lessons
* My code: https://github.com/hanhanwu/Hanhan_Data_Science_Practice/blob/master/Better4Industry/explore_features_fastai.ipynb
* I thought these exploration needs fastai, and spent lots of time to install it, then it turned out, not have to use fastai library
* The methods here are majorly to explore features from another perspective
  * Feature contribution for Out Of Bag score
  * Feature correlation in hieiarchical clustered view
  * Feature contribution in each record. From above Lime, it only shows visualization, but sometimes we need to process a vunch of records, `treeinterpreter` here helps a lot.
    
## SHARED INDUSTRY EXPERIENCE
### Digital Marketing
* https://www.analyticsvidhya.com/blog/2018/12/guide-digital-marketing-analytics/
  * It majorly talks about how does Google ad and relevant products work. Also includes how does cookies help in ads.
    
[1]:https://github.com/hanhanwu/Hanhan_Data_Science_Practice/blob/master/Better4Industry/ExperienceNotes.md
[2]:https://github.com/hanhanwu/Hanhan_Data_Science_Practice/blob/master/Better4Industry/Feature_Selection_Collection/ReadMe.md#shap
