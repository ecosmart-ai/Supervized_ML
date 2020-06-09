# -*- coding: utf-8 -*-
"""
Created on Tue Jun  9 11:28:42 2020

@author: elias
"""

# # logistic classifier
from sklearn.linear_model import LogisticRegression
classifier_LR = LogisticRegression(random_state = 0  ,penalty='l2'  ) # ,penalty='l2' , C=0.0069 , C=0.0069

# #KNN
from sklearn.neighbors import KNeighborsClassifier
classifier_KNN = KNeighborsClassifier(n_neighbors = 6, metric = 'minkowski') #, p = 2


# # Fitting Kernel SVM to the Training set
from sklearn.svm import SVC
classifier_SVM = SVC(kernel = 'rbf', random_state = 0)
 
# Fitting Naive Bayes to the Training set
from sklearn.naive_bayes import GaussianNB
classifier_GNB = GaussianNB()

# Fitting Random Forest Classification to the Training set
from sklearn.ensemble import RandomForestClassifier
classifier_RFC = RandomForestClassifier(n_estimators = 10, criterion = 'entropy', random_state = 0)

#fitting XGBOOST
from xgboost import XGBClassifier
classifier_XGBOOST = XGBClassifier(random_state = 0,n_jobs=-1)

List_Model = ['LR','KNN','GNB' ,'SVM-rfb' ,'RandomForest','XGBOOST'] #
classifers = [classifier_LR,classifier_KNN,classifier_GNB  ,classifier_SVM, classifier_RFC,classifier_XGBOOST]


from sklearn.metrics import confusion_matrix,accuracy_score,precision_score,recall_score,f1_score,roc_auc_score


for classifer in classifers:
    classifer.fit(X_train, y_train)
    y_pred = classifer.predict(X_test)
    cm = confusion_matrix(y_test, y_pred)
    acc= accuracy_score(y_test, y_pred )
    precision = precision_score(y_test, y_pred, average='weighted')
    recall = recall_score(y_test, y_pred, average='weighted')
    f1score =  f1_score(y_test, y_pred, average='weighted')
    roc =  roc_auc_score(y_test, y_pred, average='weighted')
    result=result.append({'model' : List_Model[i] , 'accuracy' : acc ,'precision' : precision , 'recall' : recall , 'F1score' : f1score  , 'ROC' : roc  } , ignore_index=True)




# grid search
from sklearn.model_selection import GridSearchCV

# LR Grid Search
grid = {"C":np.logspace(-3,3,7), "penalty":["l2"]}# l1 lasso l2 ridge
logreg = LogisticRegression(random_state=88 )
classifer = GridSearchCV(logreg,grid,cv=5,scoring='f1' )

#KNN Grid Search
grid ={'n_neighbors': np.arange(9,13),'weights':['uniform','distance'],'metric':['minkowski', 'euclidean','manhattan'] } #, 'distance', 'euclidean','manhattan',
KNN_Class = KNeighborsClassifier( )
classifer = GridSearchCV(KNN_Class,grid,cv=5,verbose=1,n_jobs=-1,scoring='f1' )

#GNB no grid search because no hyper parameter

#SVM Grid Search


 
grid ={'C': np.logspace(0,3,4) , 'gamma' : np.logspace(-4,0,5)}
classifier_SVM = SVC(kernel = 'rbf') 
classifer = GridSearchCV(classifier_SVM,grid,cv=8,verbose=1,n_jobs=-1,scoring='f1' )   

classifer.fit(X_train, y_train)
print("tuned hpyerparameters :(best parameters) ", classifer.best_params_)