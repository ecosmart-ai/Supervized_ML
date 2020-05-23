#!/usr/bin/env python
# coding: utf-8



import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

get_ipython().run_line_magic('matplotlib', 'inline')




df =pd.read_csv('data_salary.csv')
df.head()




#sns.heatmap(pd.pivot_table(df , values='id', index=['education'],columns=['education_num'], aggfunc=np.count_nonzero))
df.groupby(['education_num','education'])['id'].count()
country_map = {'United-States': 'USA','Mexico': 'Americas','?': 'Other','Philippines': 'Asia','Germany': 'Europe',
               'Canada': 'Americas','Puerto-Rico': 'Americas','El-Salvador': 'Americas','India': 'Asia','Cuba': 'Americas',
               'England': 'Europe','Jamaica': 'Americas','South': 'South','China': 'Asia','Italy': 'Europe',
               'Dominican-Republic': 'Americas','Vietnam': 'Asia','Guatemala': 'Americas','Columbia': 'Americas',
               'Japan': 'Asia','Poland': 'Europe','Taiwan': 'Asia','Haiti': 'Americas','Iran': 'Asia','Portugal': 'Europe',
               'Nicaragua': 'Americas','Peru': 'Americas','Greece': 'Europe','France': 'Europe','Ecuador': 'Americas',
               'Ireland': 'Europe','Hong': 'Asia','Laos': 'Asia','Trinadad&Tobago': 'Americas','Cambodia': 'Asia',
               'Thailand': 'Asia','Yugoslavia': 'Europe','Outlying-US(Guam-USVI-etc)': 'USA','Honduras': 'Americas',
               'Hungary': 'Europe','Scotland': 'Europe','Holand-Netherlands': 'Europe'}
country_map_num = {'USA':1, 'Americas':2 , 'Asia':3 , 'Europe':4 , 'Others':5}
race_map = {'White':'White', 'W hite':'White' ,  'Black': 'Black' , 'Asian-Pac-Islander':'Asian' , 'Amer-Indian-Eskimo': 'Amerindian' , 'Other' : 'other'}
race_num={'White':1 ,'Black':2 ,'Asian':3, 'Amerindian':4,'other':5}
workclass_map={'Private':'Priv','Self-emp-not-inc':'Self_emp', 'Self-emp-inc':'Self_emp', 'Local-gov':'gov','State-gov':'gov' ,'Federal-gov':'gov','?':'other', 'Without-pay':'nopay','Never-worked':'nopay'}
workclass_map_num={'Priv':1,'Self_emp':2,'gov':3 ,'other':4 , 'nopay':5} 
Highpay_map = {'<=50K':0, '>50K':1}
sex_num = {'Male':1,'Female':0}




education_map={1:1,2:1,3:1,4:2,5:2, 6:2,7:2 , 8:2 , 9:3,10:4,11:5,12:5,13:6,14:7,15:7,16:7}
df1=df.copy()
df1['edu_num']=df1['education_num'].map(education_map)
df1['workclass']=df1['workclass'].map(workclass_map)
df1['country']=df1['native_country'].map(country_map)
df1['race']=df1['race'].map(race_map)
df1['over_50k']=df1['over_50k'].map(Highpay_map)
df1=df1.drop(['education_num','education','id','native_country'],axis=1)
df1=df1.dropna()


df2=df1.copy()
df2['workclass']=df2['workclass'].map(workclass_map_num)
df2['country']=df2['country'].map(country_map_num)
df2['race']=df2['race'].map(race_num)
df2['sex']=df2['sex'].map(sex_num)
df2=df2.drop(['marital_status','occupation','relationship','race'],axis=1)
df2.info()



df2.over_50k.value_counts()





sns.set(style="whitegrid")
plt.figure(figsize=(10,6))
#g= sns.FacetGrid(df1,col='country', row='sex')
#g= g.map(plt.hist,'age', bins=20)
#sns.kdeplot(df['age'],shade=True )
#sns.barplot(data=df1, x='edu_num',y='age')
#sns.pairplot(df2,)

plt.hist(df2[df2.over_50k==0]['age'],bins=30,stacked=True,color='b',alpha=0.5,label='<=50k$')
plt.hist(df2[df2.over_50k==1]['age'],bins=30,stacked=True,color='r',alpha=0.5,label='>50k$')# age=-1 are young unemployed so salary is considered always 0

plt.legend()
plt.xlabel('Age')



df1=df1[df1['age']!=-1]
df1.info()




X=df1.drop(['over_50k'],axis=1)
y=df1['over_50k']

feat_col=['workclass','marital_status','occupation','relationship','race','sex','country']

X=pd.get_dummies(df1,columns=feat_col,drop_first=True)
from sklearn.decomposition import PCA
pca = PCA(n_components=12)
X = pca.fit_transform(X)
explained_variance = pca.explained_variance_ratio_
print(np.round(explained_variance*100,decimals=1))




from sklearn.model_selection import train_test_split
X_train,X_test, y_train, y_test = train_test_split(X,y, test_size=0.3, random_state=42)




# downsampling
from collections import Counter
from imblearn.under_sampling import RandomUnderSampler
rus = RandomUnderSampler(random_state=42)
X_train, y_train = rus.fit_resample(X_train, y_train)

from sklearn.linear_model import LogisticRegression
lr= LogisticRegression()
lr.fit(X_train,y_train)
y_pred=lr.predict(X_test)

from sklearn.metrics import classification_report , confusion_matrix , roc_auc_score

print('Report',classification_report(y_test, y_pred) )
print('cm',confusion_matrix(y_test, y_pred) )
print('auc',roc_auc_score(y_test, y_pred) )





