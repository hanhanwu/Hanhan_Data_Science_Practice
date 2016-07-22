import pandas as pd

X_train = pd.read_csv('[your file path]/X_train.csv')
Y_train = pd.read_csv('[your file path]/Y_train.csv')

X_test = pd.read_csv('[your file path]/X_test.csv')
Y_test = pd.read_csv('[your file path]/Y_test.csv')


print (X_train.head())
print
print (X_test.head())


# check data types
X_train.dtypes


%matplotlib inline

# Apply Feature Scaling on continuous variables so that they can be compared on the same ground
import matplotlib.pyplot as plt

p = X_train[X_train.dtypes[(X_train.dtypes=="float64")|(X_train.dtypes=="int64")]
                        .index.values].hist(figsize=[11,11])
