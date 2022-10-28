# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
from sklearn.preprocessing import OrdinalEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.ensemble import RandomForestClassifier


# read data
train_df = pd.read_csv('train.csv')
test_df = pd.read_csv('test.csv')

train_df = train_df.drop(['Ticket', 'Cabin'], axis=1)
test_df = test_df.drop(['Ticket', 'Cabin'], axis=1)

# data clean
train_df['Embarked']= train_df['Embarked'].fillna('C')
age_mean = round(train_df['Age'].mean())
train_df['Age'] = train_df['Age'].fillna(age_mean)
age_mean = round(test_df['Age'].mean())
test_df['Age'] = test_df['Age'].fillna(age_mean)
fare_mean = round(test_df['Fare'].mean())
test_df['Fare'] = test_df['Fare'].fillna(fare_mean)

encoder= OrdinalEncoder(categories=[['male', 'female']])
encoder.fit(train_df[['Sex']])
train_df['Sex_enc'] = encoder.transform(train_df[['Sex']])
encoder= OrdinalEncoder(categories=[['male', 'female']])
encoder.fit(test_df[['Sex']])
test_df['Sex_enc'] = encoder.transform(test_df[['Sex']])

train_df['Sex_enc'] = train_df['Sex_enc'].astype(int)
test_df['Sex_enc'] = test_df['Sex_enc'].astype(int)

combine = [train_df, test_df]
for dataset in combine:
    dataset['Title'] = dataset.Name.str.extract(' ([A-Za-z]+)\.', expand=False)
pd.crosstab(train_df['Title'], train_df['Sex'])
for dataset in combine:
    dataset['Title'] = dataset['Title'].replace(['Lady', 'Countess','Capt', 'Col',
                                                 'Don', 'Dr', 'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona'], 'Rare')

    dataset['Title'] = dataset['Title'].replace('Mlle', 'Miss')
    dataset['Title'] = dataset['Title'].replace('Ms', 'Miss')
    dataset['Title'] = dataset['Title'].replace('Mme', 'Mrs')
    
train_df[['Title', 'Survived']].groupby(['Title'], as_index=False).mean()

title_mapping = {"Mr": 1, "Miss": 2, "Mrs": 3, "Master": 4, "Rare": 5}
for dataset in combine:
    dataset['Title'] = dataset['Title'].map(title_mapping)
    #dataset['Title'] = dataset['Title'].fillna(0)

train_df = train_df.drop(['Name', 'PassengerId'], axis=1)
test_df = test_df.drop(['Name'], axis=1)
train_df = train_df.drop(['Sex'], axis=1)
test_df = test_df.drop(['Sex'], axis=1)

combine= [train_df, test_df]
for dataset in combine:
    dataset['Embarked'] = dataset['Embarked'].map( {'S': 0, 'C': 1, 'Q': 2} ).astype(int)

# Base
X_train = train_df.drop('Survived', axis=1)
Y_train = train_df['Survived']
X_test = test_df.drop('PassengerId', axis=1).copy()
print("X_train shape: {0}".format(X_train.shape))
print("Y_train shape: {0}".format(Y_train.shape))
print("X_test shape: {0}".format(X_test.shape))


# Logistic Regression
logistic_R = LogisticRegression(max_iter=1000)
logistic_R.fit(X_train, Y_train)
Y_pred = logistic_R.predict(X_test)
accuracy_LR = round(logistic_R.score(X_train, Y_train) * 100, 2)

feature_cols = ['Pclass', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked',
       'Sex_enc', 'Title']
x = train_df[feature_cols]
y = train_df.Survived
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size= 0.25, random_state=0)
logreg = LogisticRegression(max_iter=1000)
logreg.fit(x_train, y_train)
y_pred = logreg.predict(x_test)
print("Exactitud", metrics.accuracy_score(y_test, y_pred))

Y_pred = logreg.predict(X_test)
submission = pd.DataFrame({
    'PassengerId': test_df['PassengerId'],
    'Survived': Y_pred
})
submission.to_csv('submission.csv', index=False)

