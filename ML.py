#!/usr/bin/env python3
import pandas as pd 
from sklearn.metrics import accuracy_score 
from sklearn.metrics import recall_score  
from sklearn.naive_bayes import MultinomialNB
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn import metrics


def decisiontree(x_train, x_test, y_train, y_test):
    clf = DecisionTreeClassifier(criterion="entropy", max_depth=3)
    # Train Decision Tree Classifer
    clf = clf.fit(x_train, y_train)
    #Predict the response for test dataset
    y_pred = clf.predict(x_test)
    print("Accuracy:",metrics.accuracy_score(y_test, y_pred))


def Multinomial(x_train, x_test, y_train, y_test):
    modelM = MultinomialNB().fit(x_train, y_train)
    predM = modelM.predict(x_test)
    accuracy = accuracy_score(y_test, predM)
    recall = recall_score(y_test, predM, average='binary')
    print('The Recall of the Multinomial model is', recall)
    print('Accuracy %.2f' % (accuracy*100))
    #print("score of Naive Bayes algo is :" , score)

def GNB(x_train, x_test, y_train, y_test):
    modelM = GaussianNB().fit(x_train, y_train)
    predM = modelM.predict(x_test)
    accuracy = accuracy_score(y_test, predM)
    pd.crosstab(y_test, predM, rownames=['Actual'], colnames=['Predicted'], margins=True)

    recall = recall_score(y_test, predM, average='binary')
    print('The Recall of the Gaussian model is', recall)
    print('Accuracy %.2f' % (accuracy*100))
    #print("score of Naive Bayes algo is :" , score)

def GradientBoosting(x_train, x_test, y_train, y_test):
    clf = GradientBoostingClassifier()
    clf.fit(x_train, y_train)

    y_pred = clf.predict(x_test)
    print(clf.score(x_test, y_test))

    accuracy = accuracy_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred, average='binary')
    print('The Recall of the Gradient model is', recall)
    print('Accuracy %.2f' % (accuracy*100))
    
def HistGradientBoosting(x_train, x_test, y_train, y_test):
    clf = HistGradientBoostingClassifier()
    clf.fit(x_train, y_train)

    y_pred = clf.predict(x_test)
    print(clf.score(x_test, y_test))

    accuracy = accuracy_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred, average='binary')
    print('The Recall of the Gradient model is', recall)
    print('Accuracy %.2f' % (accuracy*100))




     
