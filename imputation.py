import pandas as pd
import numpy as np

#load dataset
data = pd.read_csv("C:/Users/user/Desktop/DATASETS/claimants.csv")
data.head() #  shows first 5 rows of dataframe
data.columns
data.describe() #describe tells about the distribution like min,max,mean,median,Q1,Q3.
#checking whether data is having any NA/null values
data.isna().sum() #having 290 na values

# ATTORNEY,CLMSEX,CLMINSUR,SEATBELT are categorical/discrete variables,so MOST FREQUENT VALUE will be used to fill NA values.
from sklearn.impute import SimpleImputer

mostfrequent_imputer = SimpleImputer(missing_values=np.nan, strategy = 'most_frequent')
data['CLMSEX'] = pd.DataFrame(mostfrequent_imputer.fit_transform(data[['CLMSEX']]))
data['CLMSEX'].isnull().sum() #na values replaced by most frequent value

mostfrequent_imputer = SimpleImputer(missing_values=np.nan, strategy = 'most_frequent')
data['CLMINSUR'] = pd.DataFrame(mostfrequent_imputer.fit_transform(data[['CLMINSUR']]))
data['CLMINSUR'].isnull().sum() #na values replaced by most frequent value

mostfrequent_imputer = SimpleImputer(missing_values=np.nan, strategy = 'most_frequent')
data['SEATBELT'] = pd.DataFrame(mostfrequent_imputer.fit_transform(data[['SEATBELT']]))
data['SEATBELT'].isnull().sum() #na values replaced by most frequent value

mean_imputer = SimpleImputer(missing_values=np.nan, strategy = 'mean')
data['CLMAGE'] = pd.DataFrame(mean_imputer.fit_transform(data[['CLMAGE']]))
data['CLMAGE'].isnull().sum() #na values replaced by mmean value


data.isna().sum() #data has no NA values
