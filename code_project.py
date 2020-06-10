# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LogisticRegression 
from sklearn.linear_model import SGDClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedKFold
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

import pandas as pd
train = pd.read_csv('/kaggle/input/titanic/train.csv')
train.head()
test = pd.read_csv('/kaggle/input/titanic/test.csv')
test.head()
test.info()
train.isnull().sum()
train1 = train.drop(['Cabin','Ticket','PassengerId','Name'], axis = 1)
test1 = test.drop(['Cabin','Ticket','PassengerId','Name'], axis = 1)

train1['SP'] = train1['SibSp'] + train1['Parch'] + 1
train1.drop(['SibSp','Parch'], axis = 1,inplace = True)

test1['SP'] = test1['SibSp'] + test1['Parch'] + 1
test1.drop(['SibSp','Parch'], axis = 1,inplace = True)

train1['Embarked'].fillna('S',inplace = True)
train1['Age'].fillna(train1.Age.mean(), inplace = True)
Embarked_dummy = pd.get_dummies(train1['Embarked'],drop_first = True)

Embarked_dummy = pd.get_dummies(train1['Embarked'],drop_first = True)
Sex = pd.get_dummies(train1['Sex'],drop_first = True)
train2 = pd.concat([train1,Embarked_dummy,Sex], axis = 1)
train2.rename(columns={'male':'Sex1'}, inplace = True)
train2.drop(['Embarked','Sex'], axis = 1, inplace = True)
train2.head()

X = test1[(test1['Sex']== 'male') & (test1['SP'] == 1) & (test1['Pclass'] == 3) & (test1['Embarked'] == 'S')]
v = X['Fare'].mean()
test1['Fare'].fillna(v,inplace = True)
test1['Age'].fillna(test1.Age.mean(),inplace = True)
test1.info()

Embarked_dummy2 = pd.get_dummies(test1['Embarked'],drop_first = True)
sex = pd.get_dummies(test1['Sex'],drop_first = True)
test2 = pd.concat([test1,Embarked_dummy2,sex], axis = 1)
test2.rename(columns={'male':'sex'}, inplace = True)
test2.drop(['Embarked','Sex'], axis = 1, inplace = True)
test2.head()

X = train2.drop(['Survived'], axis = 1)
Y = train2.Survived.copy()

kfolds = StratifiedKFold(n_splits=4, random_state = 2)
log_reg = LogisticRegression()
accuracy = 0
z = 0
for train_index,test_index in kfolds.split(X,Y):
    Xtrain = X.loc[train_index]
    Ytrain = Y.loc[train_index]
    Xtest = X.loc[test_index]
    Ytest = Y.loc[test_index]
    
    
    log_reg.fit(Xtrain,Ytrain)
    Ypredict = log_reg.predict(Xtest)
    accuracy += accuracy_score(Ytest,Ypredict)
    z += 1

accuracy/=4    
print(accuracy)

output = log_reg.predict(test2)
output1 = [pd.DataFrame(test['PassengerId']) , pd.DataFrame(output,columns = ['Survived'])]
Output_Final = pd.concat(output1, axis = 1)

Output_Final.to_csv('submission.csv')