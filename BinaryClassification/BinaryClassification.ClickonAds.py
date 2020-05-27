#!/usr/bin/env python
# coding: utf-8

# ___
# 
# ___
# # classification
# 
# In this project we will be working with  advertising data set, indicating whether or not a particular internet user clicked on an Advertisement. We will try to create a model that will predict whether or not they will click on an ad based off the features of that user.
# 
# This data set contains the following features:
# 
# * 'Daily Time Spent on Site': consumer time on site in minutes
# * 'Age': cutomer age in years
# * 'Area Income': Avg. Income of geographical area of consumer
# * 'Daily Internet Usage': Avg. minutes a day consumer is on the internet
# * 'Ad Topic Line': Headline of the advertisement
# * 'City': City of consumer
# * 'Male': Whether or not consumer was male
# * 'Country': Country of consumer
# * 'Timestamp': Time at which consumer clicked on Ad or closed window
# * 'Clicked on Ad': 0 or 1 indicated clicking on Ad
# 
# ## Import Libraries
# 
# **Import a few libraries you think you'll need (Or just import them as you go along!)**

# In[3]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')
sns.set_style('whitegrid')



ad_data=pd.read_csv('advertising.csv')


ad_data.head()
ad_data.describe()
ad_data.info()
ad_data.isna().sum()



plt.hist(x='Age',data=ad_data,bins=30)
plt.xlabel('Age')

sns.jointplot(x='Age', y = 'Area Income', data=ad_data )
sns.jointplot(x='Age', y = 'Daily Time Spent on Site', data=ad_data , kind='kde' , color='r')
sns.jointplot(x='Daily Time Spent on Site', y = 'Daily Internet Usage', data=ad_data  , color='g')
sns.pairplot(ad_data,hue='Clicked on Ad')

# create the model

from sklearn.model_selection import train_test_split
X=ad_data [ ["Daily Time Spent on Site","Age","Area Income" ,"Daily Internet Usage","Male"]]
y=ad_data.iloc[:,-1]
X_train , X_test, y_train,y_test = train_test_split(X,y,test_size=0.3, random_state=101)


from sklearn.linear_model import LogisticRegression
lr= LogisticRegression()
lr.fit(X_train,y_train)
y_pred=lr.predict(X_test)


from sklearn.metrics import classification_report
print (classification_report(y_test,y_pred))

