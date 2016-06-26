# I'm using Spark Cloud Community Edition, sicne my own machine cannot have the right numpy for pandas...
# So, in this code, so features could only be used in Spark Cloud Python Notebook
# Try pandas :)

# cell 1 - load the data (I upload the .csv into Spark Cloud first)
import pandas as pd
import numpy as np

## The path here is the .csv file path in HDFS
pdata = sqlContext.read.format('csv').load("/FileStore/tables/[file name in HDFS]", 
                                       index_col="ID", header =True).toPandas()
                                       
                                       
# cell 2 - Bollean Indexing
pdata.loc[(pdata["Gender"]=="Female") & (pdata["Salary_Account"]=="ICICI Bank") & (pdata["Mobile_Verified"]=="Y"), 
["Gender", "Salary_Account", "Mobile_Verified"]]


# cell 3 - apply function, similar to R apply()
def get_missing_data(x):
  return sum(x.isnull())

print "find missing data for each column:"
print pdata.apply(get_missing_data, axis = 0)

print "find missing data for each row:"
print pdata.apply(get_missing_data, axis = 1)


# cell 4 - fillna(), updating missing values with the overall mean/mode/median of the column
from scipy.stats import mode

# check the mode
mode(pdata['Gender'])[0][0] 

pdata['Gender'].fillna(mode(pdata['Gender'])[0][0], inplace=True)
pdata.apply(get_missing_data, axis=0)


# cell 5 - create Excel style pivot table
# check data type first
pdata.dtypes

# convert Monthly_Income into numerical data
pdata['Monthly_Income'] = pdata['Monthly_Income'].astype(float)
pdata.dtypes

pivot_t = pdata.pivot_table(values=['Monthly_Income'], index=['Gender', 'Mobile_Verified', 'Device_Type'], aggfunc = np.mean)
print pivot_t


# cell 6 - MUltiple Indexing
## I like this, only iterate rows with Monthly_Income as null
for i, r in pdata.loc[pdata['Monthly_Income'].isnull(),:].iterrows():
  index_list = tuple([r(['Gender']), r(['Mobile_Verified']), r(['Device_Type'])])
  pdata.loc[i, 'Monthly_Income'] = pivot_t.loc[index_list].values[0]   # using multiple index to locate data
  
print pdata.apply(get_missing_data, axis=0)


# cell 7 - cross tab
print pd.crosstab(pdata['Gender'], pdata['Mobile_Verified'], margins=True)
print

def get_percentage(ser):
  return ser/float(ser[-1])

print pd.crosstab(pdata['Gender'], pdata['Mobile_Verified'], margins=True).apply(get_percentage, axis=1)


# cell 8 - data merging
people_rate = pd.DataFrame([200, 400], index=['Mobile', 'Web-browser'], columns=['people_rate'])
people_rate

data_merge = pdata.merge(right=people_rate, how='inner', left_on='Device_Type', right_index=True, sort=False)
data_merge.pivot_table(values=['Monthly_Income'], index=['Device_Type', 'people_rate'], aggfunc = len)
