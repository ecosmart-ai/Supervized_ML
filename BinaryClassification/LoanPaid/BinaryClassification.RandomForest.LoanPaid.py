

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')



loans=pd.read_csv('loan_data.csv')
loans.info()
loans.isna().sum()
loans.head()
loans.describe()


#  a histogram of two FICO distributions on top of each other, one for each credit.policy outcome



plt.style.use('ggplot')
plt.figure(figsize=(10,6))
plt.legend()
plt.xlabel('FICO')
plt.hist(loans[loans['credit.policy']==1]['fico'],bins=30,alpha=0.5,color='blue',  label='Credit.Policy=1') 
plt.hist(loans[loans['credit.policy']==0]['fico'],bins=30,alpha=0.5,color='red',  label='Credit.Policy=0') 






# a histogram of two FICO distributions on top of each other, one for each not.fully.paid column.**



plt.figure(figsize=(10,6))

plt.hist(loans[loans['not.fully.paid']==1]['fico'],bins=30,alpha=0.5,color='blue',  label='not.fully.paid=1') 
plt.hist(loans[loans['not.fully.paid']==0]['fico'],bins=30,alpha=0.5,color='red',  label='not.fully.paid=0') 
plt.legend()
plt.xlabel('FICO')



# countplot using seaborn showing the counts of loans by purpose, with the color hue defined by not.fully.paid



plt.figure(figsize=(20,10))
sns.countplot(x='purpose',data=loans,hue='not.fully.paid' )







#  trend between FICO score and interest rate. 


sns.color_palette("viridis")
sns.jointplot(x='fico',y='int.rate',data=loans,color='purple')


# trend difference between not.fully.paid and credit.policy. Check the documentation for lmplot()



sns.lmplot(x="fico", y="int.rate", col="not.fully.paid", hue='credit.policy', data=loans)



loans.info()

 
#feature engineering
cat_feats= ['purpose']
final_data = pd.get_dummies(loans,columns=cat_feats,drop_first=True)
final_data.info()


#build classification decision tree


from sklearn.model_selection import train_test_split
X = final_data.drop('not.fully.paid',axis=1)
y = final_data['not.fully.paid']
X_train , X_test,y_train, y_test = train_test_split(X,y, test_size=0.3, random_state=42)



from sklearn.tree import DecisionTreeClassifier
dtree= DecisionTreeClassifier()
dtree.fit(X_train,y_train)
y_pred=dtree.predict(X_test)

from sklearn.metrics import classification_report , confusion_matrix
print('classification report:')
print(classification_report(y_test,y_pred))

print('confusion matrix:')
print(confusion_matrix(y_test,y_pred))



# Training the Random Forest model

from sklearn.ensemble import RandomForestClassifier
rf = RandomForestClassifier(n_estimators=600)
rf.fit(X_train, y_train)
y_pred=rf.predict(X_test)
print(classification_report(y_pred,y_test))


print('confusion matrix:')
print(confusion_matrix(y_test,y_pred))

#recall is low for the minority class : feature engineering is needed to rebalance the data and improve the recall, F1 socre and auc roc
