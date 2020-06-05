# predict avocado prices

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import random
import seaborn as sns
from fbprophet import Prophet



df=pd.read_csv('avocado.csv')
df.head()


# EDA
df = df.sort_values('Date')
plt.figure(figsize=(12,5))
plt.plot(df['Date'],df['AveragePrice'])


plt.figure(figsize=(25,12))
sns.countplot(data=df, x='region')
plt.xticks(rotation=45)


sns.countplot(data=df, x='year')

df_total= df[['Date','AveragePrice']]



# prophet prediction
prophet_df=df_total.rename(columns={'Date':'ds', 'AveragePrice':'y'})
prophet_df
P= Prophet()
P.fit(prophet_df)


# Forecast the future
future = P.make_future_dataframe(periods=365)
forecast = P.predict(future)

figure = P.plot(forecast, xlabel="Date",ylabel="Sales")
figure = P.plot_components(forecast)



# forecast on a period of time
df_sample = df[df['region']=='West']
df_sample = df_sample.sort_values('Date')


plt.figure(figsize=(12,3))
plt.plot(df_sample['Date'],df_sample['AveragePrice'])



df_sample_Prophet =df_sample.rename(columns={'Date':'ds','AveragePrice':'y'})
P2= Prophet()
P2.fit(df_sample_Prophet)


future = P2.make_future_dataframe(periods=365)
forecast = P2.predict(future)
figure= P2.plot(forecast,xlabel='Date',ylabel='forecast')
figure = P2.plot_components(forecast)

