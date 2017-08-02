import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt 

from sklearn import preprocessing 
from sklearn.metrics import mean_squared_error 

from tpot import TPOTRegressor


train = pd.read_csv("train_GA.csv")
test = pd.read_csv("test_GA.csv")

pd.isnull(train).sum() > 0
pd.isnull(test).sum() > 0


# TO BE CONTINUED....
