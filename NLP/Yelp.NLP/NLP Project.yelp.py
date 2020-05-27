

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

get_ipython().run_line_magic('matplotlib', 'inline')

yelp = pd.read_csv('yelp.csv')
yelp.head()
yelp.info()
yelp.describe()


yelp['text length'] = yelp['text'].apply(lambda x: len(x))
yelp.head()


g = sns.FacetGrid(data=yelp,col='stars')
g=g.map(plt.hist,'text length',bins=20)

sns.boxplot( data=yelp  ,x='stars' ,y='text length' )
plt.xlabel('stars')
sns.countplot(data=yelp, x='stars' , palette='rainbow')


stars=yelp.groupby(['stars']).mean()
sns.heatmap((stars.corr()),annot=True , cmap='coolwarm')

yelp_class = yelp[(yelp['stars']==1 )| (yelp['stars']==5)]
yelp_class.head()

X=yelp_class['text']
y=yelp_class['stars']


from sklearn.feature_extraction.text import CountVectorizer
cv= CountVectorizer()


X=cv.fit_transform(X)




from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.3, random_state=101)
from sklearn.naive_bayes import MultinomialNB

MNB = MultinomialNB()
MNB.fit(X_train, y_train)
y_pred = MNB.predict(X_test)

from sklearn.metrics import classification_report , confusion_matrix
print(confusion_matrix(y_test,y_pred))
print (classification_report(y_test,y_pred))

from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
pipeline= Pipeline([
    ('bow',CountVectorizer()),
    #('tfidf',TfidfTransformer()),
    ('classifier',MultinomialNB()),
])

 


X=yelp_class['text']
y=yelp_class['stars']
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.3, random_state=101)
pipeline.fit(X_train,y_train)


y_pred=pipeline.predict(X_test)
print (confusion_matrix(y_test,y_pred))
print (classification_report(y_test,y_pred))
