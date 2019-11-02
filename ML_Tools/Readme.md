# ML Tools

## MLFlow
* [Try Tracking UI][1]
  * `mlruns` stores the logs, by default it's under your username.
  * To find it's location, on windows you can type `dir mlruns` under your username, it will show the full path.
* [Running MLFlow Projects][2]
  * I tried to run the commands in this example, tried on both Windows, Mac, in powershell or bash, in virtual environment or locally environment, none of them works. Either it kept asking me for `conda init` or it's showing brokwn numpy problem... I tried to fix but only got same error or even worse. Better to set this problem a side at this moment.
* [Saving & Serving Model][3]
  * I tried this function too, at least it's runable. But I don't like the log experience. Each param is stored in a single file with only 1 value in the file. `log_model` didn't really save any model....


[1]:https://github.com/hanhanwu/Hanhan_Data_Science_Practice/blob/master/ML_Tools/try_mlflow_trackingUI.ipynb
[2]:https://github.com/mlflow/mlflow/blob/master/docs/source/quickstart.rst#running-mlflow-projects
[3]:https://github.com/mlflow/mlflow/blob/master/docs/source/quickstart.rst#saving-and-serving-models
