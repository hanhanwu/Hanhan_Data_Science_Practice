import pandas as pd
from keras.layers import LSTM
import matplotlib.pyplot as plt
from datetime import datetime

def date_parser(x):
    return datetime.strptime('190'+x, '%Y-%m')
    
series = pd.read_csv("shampoo_sales.csv", header=0, parse_dates=[0], index_col=0, squeeze=True, date_parser=date_parser)
series.head()
