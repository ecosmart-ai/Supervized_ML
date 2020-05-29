


import pandas as pd


data_info = pd.read_csv('../DATA/lending_club_info.csv',index_col='LoanStatNew')



print(data_info.loc['revol_util']['Description'])



def feat_info(col_name):
    print(data_info.loc[col_name]['Description'])


feat_info('mort_acc')


# ## Loading the data and other imports


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# might be needed depending on your version of Jupyter
get_ipython().run_line_magic('matplotlib', 'inline')




df = pd.read_csv('../DATA/lending_club_loan_two.csv')




df.info()





sns.countplot(x='loan_status',data=df)



plt.figure(figsize=(12,6))
plt.hist(df.loan_amnt,bins=50)
plt.xlabel('loan amount')



df.corr()




plt.figure(figsize=(10,10))
sns.heatmap(df.corr(),annot=True,cmap='YlGnBu')





# almost perfect correlation with the "installment" feature.


df[['loan_amnt','installment']].describe()
sns.scatterplot(data=df,x='loan_amnt',y='installment')





sns.boxplot(data=df,x='loan_status',y='loan_amnt')




df.groupby('loan_status')['loan_amnt'].describe()


gra=df['grade'].unique().tolist()
subgra=df['sub_grade'].unique().tolist()
subgra.sort()


sns.countplot(data=df, x='grade' , hue='loan_status',order=gra)



plt.figure(figsize=(12,6))
sns.countplot(data=df, x='sub_grade',order=subgra)




subgra=df[(df['grade']=='F')| (df['grade']=='G')]['sub_grade'].unique().tolist()
subgra.sort()
plt.figure(figsize=(12,6))
sns.countplot(data=df[(df['grade']=='F')| (df['grade']=='G')], x='sub_grade' ,order=subgra,hue='loan_status')



df['loan_repaid'] = df['loan_status'].map({'Fully Paid':1,'Charged Off':0})
df[['loan_repaid','loan_status']]



df.corr()['loan_repaid'].sort_values().drop('loan_repaid').plot(kind='bar')






# # Missing Data


len(df)



df.isnull().sum()*100/len(df)


df['emp_title'].nunique()
df['emp_title'].value_counts()


#  Realistically there are too many unique job titles to try to convert this to a dummy variable feature. Let's remove that emp_title column.**


df=df.drop('emp_title',axis=1)






emp_tit_order=df['emp_length'].dropna().unique().tolist()
emp_tit_order.sort()



emp_tit_order=['< 1 year','1 year', '2 years','3 years','4 years','5 years','6 years','7 years','8 years','9 years','10+ years']



plt.figure(figsize=(12,6))
sns.countplot(data=df, x='emp_length' ,order=emp_tit_order )


plt.figure(figsize=(12,6))
sns.countplot(data=df, x='emp_length' ,order=emp_tit_order,hue='loan_status')


loan_ratio=(1-df.dropna().groupby('emp_length')['loan_repaid'].sum()/df.dropna().groupby('emp_length')['loan_repaid'].count())



loan_ratio.plot(kind='bar')


df=df.drop('emp_length',axis=1)


df.isnull().sum()

df['title'].head(10)


df=df.drop('title',axis=1)



feat_info('mort_acc')



df['mort_acc'].value_counts()
df.corr()['mort_acc'].sort_values()
mean_mort_acc_by_totalacc= df.groupby('total_acc')['mort_acc'].mean()
df['mort_acc']=df.apply(lambda x : mean_mort_acc_by_totalacc[x['total_acc']] if np.isnan(x['mort_acc'])  else x['mort_acc'] ,axis=1)

df.isnull().sum()
df=df.dropna()



df.select_dtypes(['object']).columns
df['term'] = df['term'].apply(lambda term: int(term[:3]))
df=df.drop(['grade' ],axis=1)
df=df.drop(['sub_grade'],axis=1)



dummies = pd.get_dummies(df[['verification_status', 'application_type','initial_list_status','purpose' ]],drop_first=True)
df = df.drop(['verification_status', 'application_type','initial_list_status','purpose'],axis=1)
df = pd.concat([df,dummies],axis=1)




df['home_ownership'].value_counts()
df['home_ownership']=df['home_ownership'].replace(['NONE', 'ANY'], 'OTHER')
dummies = pd.get_dummies(df['home_ownership'],drop_first=True)
df = df.drop('home_ownership',axis=1)
df = pd.concat([df,dummies],axis=1)



df['zip_code']=df['address'].apply(lambda x: x[-5::])
df['zip_code'].value_counts()


dummies = pd.get_dummies(df['zip_code'],drop_first=True)
df = df.drop(['zip_code','address'],axis=1)
df = pd.concat([df,dummies],axis=1)

df.select_dtypes(['object']).columns
df['earliest_cr_year']=df['earliest_cr_line'].apply(lambda x: int(x[-4::]))
df= df.drop(['earliest_cr_line'],axis=1)


#train the model

from sklearn.model_selection import train_test_split
df= df.drop(['loan_status'],axis=1)
X= df.drop(['loan_repaid'],axis=1).values
y= df['loan_repaid'].values
print(len(df))

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=101)


from sklearn.preprocessing import MinMaxScaler
scaler= MinMaxScaler()
scaler.fit(X_train)
X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)



import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense,Dropout


model = Sequential()
model.add(Dense(units=44,activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(units=79,activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(units=39,activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(units=19,activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(units=1,activation='sigmoid'))
model.compile(loss='binary_crossentropy', optimizer='adam')




model.fit(x=X_train, y=y_train, epochs=25,batch_size=256,validation_data=(X_test, y_test), verbose=1)
from tensorflow.keras.models import load_model
model.save('full_data_project_model_test1_EZ.h5')  



model_loss = pd.DataFrame(model.history.history)
model_loss.plot()


y_pred= model.predict_classes(X_test)

from sklearn.metrics import classification_report, confusion_matrix
print (confusion_matrix(y_test,y_pred))
print (classification_report(y_test,y_pred))


# predict a random customer status

import random
random.seed(101)
random_ind = random.randint(0,len(df))

new_customer = df.drop('loan_repaid',axis=1).iloc[random_ind]
new_customer


model.predict_classes(new_customer.values.reshape(1,-1))
df['loan_repaid'].iloc[random_ind]
