import pandas as pd
import numpy as np
df = pd.read_csv("C:/Users/user/Desktop/DATASETS/OnlineRetail.csv",encoding = 'unicode_escape')
df.shape
df.describe()
df.columns
df.dtypes
#coverting unitprice and customerid from float to int
df['UnitPrice'] = df.UnitPrice.astype('int64')
df['CustomerID'] = df.CustomerID.astype('int64') #cannot convert to int as the data having NA values

from sklearn.impute import SimpleImputer
mostfrequent_imputer = SimpleImputer(missing_values=np.nan, strategy = 'most_frequent')
df['CustomerID'] = pd.DataFrame(mostfrequent_imputer.fit_transform(df[['CustomerID']]))
df['CustomerID'] = df.CustomerID.astype('int64')
#now data is coverted from float to int (typecasting is done)
 
duplicate = df.duplicated()
sum(duplicate) #5268 duplicated values are present in the data

#Removing Duplicates
df1 = df.drop_duplicates() 
df1.duplicated().sum() #duplicates have  been removed

 #EDA
#for any EDA calulation,  serial number and customer id has no use.
#EDA on unitprice,quantity
 #first moment business decision
df1.UnitPrice.mean()
df1.Quantity.mean() 
df1.UnitPrice.median()
df1.Quantity.median()
df1.UnitPrice.mode()
df1.Quantity.mode()
#second moment business decision
df1.UnitPrice.var()
df1.Quantity.var()

df1.UnitPrice.std()
df1.Quantity.std()

range_unitprice = df1.UnitPrice.max() - df1.UnitPrice.min()
range_Quantity = df1.Quantity.max() - df1.Quantity.min()

#third moment
df1.UnitPrice.skew() #completely right skewed
df1.Quantity.skew() #normal
#fourth moment
df1.UnitPrice.kurt()
df1.Quantity.kurt()

import matplotlib.pyplot as plt # used for visualization purposes 

plt.hist(df1.UnitPrice) #histogram

plt.boxplot(df1.UnitPrice) #boxplot
#having outliers

plt.hist(df1.Quantity) #histogram

plt.boxplot(df1.Quantity) #boxplot
#having outliers

#Normal Quantile-Quantile Plot
import scipy.stats as stats
import pylab

# Checking Whether data is normally distributed
stats.probplot(df1.UnitPrice, dist="norm",plot=pylab) # eliminating outliers on right gives normal line.

stats.probplot(df1.Quantity,dist="norm",plot=pylab)  #normal
