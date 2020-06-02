
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


df=pd.read_csv('Car_Purchasing_Data.csv',encoding='ISO-8859-1')
sns.pairplot(df)
df.info()


df=df.drop(['Customer Name','Customer e-mail'],axis=1)
df.info()

 
df.Country.value_counts()
 
#import country package in order to group countries per continent
import datapackage
data_url = 'https://datahub.io/JohnSnowLabs/country-and-continent-codes-list/datapackage.json'
country_package = datapackage.Package(data_url)
resources = country_package.resources
for resource in resources:
    if resource.tabular:
        country_data = pd.read_csv(resource.descriptor['path'])
ContinentCountry = country_data[['Continent_Name','Country_Name']]
country_data['Country']=country_data['Country_Name'].apply(lambda x: x.split(',')[0])
country_data=country_data.set_index('Country')
country_data=country_data.iloc[:,0].to_dict()
country_data2={'Syria':'Asia', 'Sint Maarten':'North America', 'Palestine, State of':'Asia', 'Laos':'Asia', 'Viet Nam':'Asia',
       'Kyrgyzstan':'Asia', 'Slovakia':'Europe', 'Falkland Islands':'other', 'Bouvet Island':'other',
       'Heard Island and Mcdonald Islands':'other', 'Virgin Islands, British':'North America',
       'Saint Barthélemy':'North America', 'Congo (Brazzaville)':'Africa', 'United States':'North America',
       'Virgin Islands, United States':'North America',
       'Saint Vincent and The Grenadines':'North America',
       'Bonaire, Sint Eustatius and Saba':'North America',
       'United Kingdom (Great Britain)':'Europe',
       'South Georgia and The South Sandwich Islands':'Antarctica', 'Antarctica':'Antarctica',
       'Korea, South':'Asia', 'Saint Helena, Ascension and Tristan da Cunha':'other',
       'marlal':'Africa'}

# merge two dictionaries
country_data_updated =  {**country_data, **country_data2}



df['Continent'] =df['Country'].map(country_data_updated) 
df['Continent'].value_counts() 
df=df.drop(['Country'],axis=1)       


df= df.join(pd.get_dummies(df['Continent'], drop_first=True))
df=df.drop(['Continent'],axis=1)       


#Scaling the data
X=df.drop(['Car Purchase Amount'],axis=1)
y=df ['Car Purchase Amount'].values
from sklearn.preprocessing import MinMaxScaler
scaler= MinMaxScaler()
scaler.fit(X)
X_scaled=pd.DataFrame(scaler.transform(X), columns=X.columns , index=X.index).values



#split the data
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=101)


#train the data
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense,Dropout



# build the network
model = Sequential()
model.add(Dense(12,activation='relu'))
model.add(Dense(16,activation='relu'))
model.add(Dense(16,activation='relu'))
model.add(Dense(16,activation='relu'))
model.add(Dense(1))
model.compile(optimizer='adam',loss='mse')


#train the network
model.fit(x=X_train,y=y_train,
          validation_data=(X_test,y_test),
          batch_size=10,epochs=100)



#plot the graph between training loss and validation loss
losses = pd.DataFrame(model.history.history)
losses.plot()
plt.title('Model Loss Progress During Training')
plt.ylabel('Training and Validation Loss')
plt.xlabel('Epoch Number')
plt.legend(['Training Loss','Validation Loss'])

from sklearn.metrics import mean_squared_error,mean_absolute_error,explained_variance_score
y_pred=model.predict(X_test)


# plot regressions between predictions and groundtruth
# Our predictions
plt.scatter(y_test,y_pred)

# Perfect predictions
plt.plot(y_test,y_test,'r')

 

# calculate the mean error and variance score
sns.distplot(y_test.reshape(-1,1)-y_pred )
print('mean error:', np.sqrt(mean_squared_error(y_test,y_pred)))
print('variance score:', explained_variance_score(y_test,y_pred))


 
#test a new customer expected amount of purchase
#Gender,Age, Annual Salary , Credit Card Debt, Net worth, Select country
X_test_sample=scaler.transform([[1,50,50000,10000,600000,1,0,0,0,0,0,0]])
y_predict_sample=model.predict(X_test_sample)
print('Expected Purchase Amount :',y_predict_sample)

