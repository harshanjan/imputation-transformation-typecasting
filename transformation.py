import pandas as pd
import numpy as np

#load dataset
data = pd.read_csv("C:/Users/user/Desktop/DATASETS/calories_consumed.csv")
data.head() #  shows first 5 rows of dataframe
data.columns #shows column names
data.describe() #describe tells about the distribution like min,max,mean,median,Q1,Q3.
#checking whether data is having any NA/null values
data.isna().sum() #having zero na values
data.dtypes

data.columns = ['weightgained','caloriesconsumed'] #changing col names for easy access

import matplotlib.pyplot as plt
plt.hist(data['weightgained']);plt.show() #right skewed
plt.boxplot(data.weightgained);plt.show() #no outliers

plt.hist(data.caloriesconsumed);plt.show() #right skewed
plt.boxplot(data.caloriesconsumed);plt.show() #no outliers

#Normal Quantile-Quantile Plot
import scipy.stats as stats
import pylab

# Checking Whether data is normally distributed
stats.probplot(data.weightgained, dist="norm",plot=pylab) #distributon is not normal
#transformation to make weightgained variable normal
import numpy as np
stats.probplot(np.log(data.weightgained),dist = "norm" , plot = pylab) #now the distribution is normal
#various transformations like log,exp,sqrt,reciprocal can be applied till the date is normally distributed.
stats.probplot(data.caloriesconsumed,dist="norm",plot=pylab) # distribution is normal

