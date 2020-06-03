
# # Support Vector Machines Project 
# 
# The Iris flower data set or Fisher's Iris data set is a multivariate data set introduced by Sir Ronald Fisher in the 1936 as an example of discriminant analysis. 
# 
# The data set consists of 50 samples from each of three species of Iris (Iris setosa, Iris virginica and Iris versicolor), so 150 total samples. Four features were measured from each sample: the length and the width of the sepals and petals, in centimeters.
# 
# Here's a picture of the three different Iris types:



# The Iris Setosa
from IPython.display import Image
url = 'http://upload.wikimedia.org/wikipedia/commons/5/56/Kosaciec_szczecinkowaty_Iris_setosa.jpg'
Image(url,width=300, height=300)
# The Iris Versicolor
from IPython.display import Image
url = 'http://upload.wikimedia.org/wikipedia/commons/4/41/Iris_versicolor_3.jpg'
Image(url,width=300, height=300)
# The Iris Virginica
from IPython.display import Image
url = 'http://upload.wikimedia.org/wikipedia/commons/9/9f/Iris_virginica.jpg'
Image(url,width=300, height=300)

import seaborn as sns
iris = sns.load_dataset('iris')
iris.head()
import pandas as pd
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')

#pair plot
sns.set_style('whitegrid')
sns.pairplot(iris,hue='species')

#  a kde plot of sepal_length versus sepal width for setosa species of flower.**

sns.set_palette('viridis')
sns.kdeplot(iris[iris['species']=='setosa']['sepal_width'],iris[iris['species']=='setosa']['sepal_length'],cmap="plasma", shade=True, shade_lowest=False)


# feature engineering 
X=iris.iloc[:,:-1]
sp_feat=['species']
Species_map={'setosa':1 ,'versicolor':2 , 'virginica':3}
y=iris['species'].map(Species_map)
from sklearn.model_selection import train_test_split

X_train , X_test, y_train, y_test = train_test_split(X,y,test_size=0.3,random_state=1986)


from sklearn.svm import SVC

svc= SVC()
svc.fit(X_train,y_train)
y_pred = svc.predict(X_test)
from sklearn.metrics import confusion_matrix , classification_report
print(confusion_matrix(y_test,y_pred))
print (classification_report(y_test,y_pred))



#  GridsearchCV 
from sklearn.model_selection import GridSearchCV
param_grid={'C':[0.1,1,10,100,1000],'gamma':[1,0.1,0.001,0.0001]}
grid =GridSearchCV(SVC(),param_grid=param_grid,verbose=3)
grid.fit(X_train,y_train)
grid.best_params_


y_pred=grid.predict(X_test)
print (confusion_matrix(y_pred,y_test))
print (classification_report(y_pred,y_test))



from sklearn.model_selection import cross_val_score
cross_svc = SVC()
accuracies= cross_val_score(cross_svc,X_train,y_train,cv=30)
print('average bias: ',accuracies.mean())
print('average variance : ',accuracies.std())

