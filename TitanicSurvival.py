#data analysis libraries
import numpy as np
import pandas as pd
#visualisation libraries
import matplotlib.pyplot as plt
import seaborn as sns

import warnings
warnings.filterwarnings('ignore')

#import files
train_data=pd.read_csv('train.csv')
test_data=pd.read_csv('test.csv')

train_data.head()
train_data.describe(include='all')
#data analysis
print(train_data.columns)
train_data.sample(5)
#to get the total number of null values in each column
print(pd.isnull(train_data).sum())

#data visualisation
#sex feature
sns.barplot(x="Sex", y="Survived", data=train_data)
print("Per of females survived:", train_data["Survived"][train_data["Sex"] == 'female'].value_counts(normalize = True)[1]*100)
print("Percentage of males who survived:", train_data["Survived"][train_data["Sex"]== 'male'].value_counts(normalize =True)[1]*100)

#Pclass feature
sns.barplot(x="Pclass", y="Survived", data=train_data)
print("1st class survival:",train_data["Survived"][train_data["Pclass"]==1].value_counts(normalize=True)[1]*100)
print("2nd class survival:",train_data["Survived"][train_data["Pclass"]==2].value_counts(normalize=True)[1]*100)
print("3rd class survival:",train_data["Survived"][train_data["Pclass"]==3].value_counts(normalize=True)[1]*100)

#SibSp features
sns.barplot(x="SibSp", y="Survived", data=train_data)
print("passenger with SibSp=0 survival:",train_data["Survived"][train_data["SibSp"]==0].value_counts(normalize=True)[1]*100)
print("passenger with SibSp=1 survival:",train_data["Survived"][train_data["SibSp"]==1].value_counts(normalize=True)[1]*100)
print("passenger with SibSp=2 survival:",train_data["Survived"][train_data["SibSp"]==2].value_counts(normalize=True)[1]*100)

#Parch feature
sns.barplot(x="Parch", y="Survived",data=train_data)
plt.show()
#bins:divide the entire range of values into a series of intervals.
#fillna is used to fill all the nan values as the passed value

#age feature
train_data["Age"]=train_data["Age"].fillna(-0.5)
test_data["Age"]=test_data["Age"].fillna(-0.5)
bins=[-1, 0, 5, 12, 18, 24, 35, 60, np.inf]
labels=["Unknown","Baby","Child","Teenager","Student","Young Adult","Adult","Senior"]
#use cut wen you need to segment and sort data values into bin
train_data["AgeGroup"]=pd.cut(train_data["Age"],bins,labels=labels)
test_data["AgeGroup"]=pd.cut(train_data["Age"],bins,labels=labels)
sns.barplot(x="AgeGroup",y="Survived",data=train_data)
plt.show()

#Cabin feature
train_data["cabinbool"]=(train_data["Cabin"].notnull().astype('int'))
test_data["cabinbool"]=(test_data["Cabin"].notnull().astype('int'))
print("percentage of cabinbool=1 who survived",train_data["Survived"][train_data["cabinbool"]==1].value_counts(normalize=True)[1]*100)
print("percentage of cabinbool=0 who survived",train_data["Survived"][train_data["cabinbool"]==0].value_counts(normalize=True)[1]*100)


#Cleaning data
test_data.describe(include="all")
#drop cabin feature
train_data=train_data.drop(["Cabin"],axis=1)
test_data=test_data.drop(["Cabin"],axis=1)

#drop ticket feature
train_data=train_data.drop(["Ticket"],axis=1)
test_data=test_data.drop(["Ticket"],axis=1)

#fill the missing value in embarked feature
print("no. of people embarking in southampton")
south=train_data[train_data["Embarked"]=="S"].shape[0]
print(south)
print("no. of people embarking in cherbourg")
cherbough=train_data[train_data["Embarked"]=="C"].shape[0]
print(cherbough)
print("no. of people embarking in queenstown")
queenstown=train_data[train_data["Embarked"]=="Q"].shape[0]
print(queenstown)
train_data["Embarked"].isnull().sum()
train_data=train_data.fillna({"Embarked":"S"})

#Age feature missing values
combine=[train_data,test_data]
train_data["Name"]
for dataset in combine:
    dataset["Title"]=dataset.Name.str.extract('([A-Za-z]+)\.',expand=False)
pd.crosstab(train_data["Title"],train_data["Sex"])
dataset["Title"]

for dataset in combine:
    dataset["Title"]=dataset["Title"].replace(['Lady','Capt','Col','Dr','Major', 'Rev', 'Jonkheer', 'Dona'], 'Rare')
    dataset["Title"]=dataset["Title"].replace(['Countess','Lady','Sir'],'Royal')
    dataset["Title"]=dataset["Title"].replace('Mlle', 'Miss')
    dataset["Title"]=dataset["Title"].replace('Ms', 'Miss')
    dataset["Title"]=dataset["Title"].replace('Mme', 'Mrs')
train_data[["Title","Survived"]].groupby(["Title"],as_index=False).mean()
title_mapping = {"Mr": 1, "Miss": 2, "Mrs": 3, "Master": 4, "Royal": 5, "Rare": 6}
for dataset in combine:
    dataset['Title'] = dataset['Title'].map(title_mapping)
    dataset['Title'] = dataset['Title'].fillna(0)

train_data.head()
mr_age = train_data[train_data["Title"] == 1]["AgeGroup"].mode()
miss_age = train_data[train_data["Title"] == 2]["AgeGroup"].mode()
mrs_age = train_data[train_data["Title"] == 3]["AgeGroup"].mode()
master_age = train_data[train_data["Title"] == 4]["AgeGroup"].mode()
royal_age = train_data[train_data["Title"] == 5]["AgeGroup"].mode() 
rare_age = train_data[train_data["Title"] == 6]["AgeGroup"].mode()

age_title_mapping = {1: "Young Adult", 2: "Student", 3: "Adult", 4: "Baby", 5: "Adult", 6: "Adult"}
for x in range(len(train_data["AgeGroup"])):
    if train_data["AgeGroup"][x] == "Unknown":
        train_data["AgeGroup"][x] = age_title_mapping[train_data["Title"][x]]
        
for x in range(len(test_data["AgeGroup"])):
    if test_data["AgeGroup"][x] == "Unknown":
        test_data["AgeGroup"][x] = age_title_mapping[test_data["Title"][x]]
        age_mapping = {'Baby': 1, 'Child': 2, 'Teenager': 3, 'Student': 4, 'Young Adult': 5, 'Adult': 6, 'Senior': 7}
train_data['AgeGroup'] = train_data['AgeGroup'].map(age_mapping)
test_data['AgeGroup'] = test_data['AgeGroup'].map(age_mapping)

train_data.head()

#dropping the Age feature for now, might change
train_data = train_data.drop(['Age'], axis = 1)
test_data = test_data.drop(['Age'], axis = 1)

#drop the name feature since it contains no more useful information.
train_data = train_data.drop(['Name'], axis = 1)
test_data = test_data.drop(['Name'], axis = 1)

#map each Sex value to a numerical value
sex_mapping = {"male": 0, "female": 1}
train_data['Sex'] = train_data['Sex'].map(sex_mapping)
test_data['Sex'] = test_data['Sex'].map(sex_mapping)

train_data.head()

#map each Embarked value to a numerical value
embarked_mapping = {"S": 1, "C": 2, "Q": 3}
train_data['Embarked'] = train_data['Embarked'].map(embarked_mapping)
test_data['Embarked'] = test_data['Embarked'].map(embarked_mapping)

train_data.head()
#fill in missing Fare value in test set based on mean fare for that Pclass 
for x in range(len(test_data["Fare"])):
    if pd.isnull(test_data["Fare"][x]):
        pclass = test_data["Pclass"][x] #Pclass = 3
        test_data["Fare"][x] = round(train_data[train_data["Pclass"] == pclass]["Fare"].mean(), 4)
        
#map Fare values into groups of numerical values
train_data['FareBand'] = pd.qcut(train_data['Fare'], 4, labels = [1, 2, 3, 4])
test_data['FareBand'] = pd.qcut(test_data['Fare'], 4, labels = [1, 2, 3, 4])

#drop Fare values
train_data = train_data.drop(['Fare'], axis = 1)
test_data = test_data.drop(['Fare'], axis = 1)
train_data.head()

from sklearn.model_selection import train_test_split

predictors = train_data.drop(['Survived', 'PassengerId'], axis=1)
target = train_data["Survived"]
x_train, x_val, y_train, y_val = train_test_split(predictors, target, test_size = 0.22, random_state = 0)
# Gaussian Naive Bayes
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score

gaussian = GaussianNB()
gaussian.fit(x_train, y_train)
y_pred = gaussian.predict(x_val)
acc_gaussian = round(accuracy_score(y_pred, y_val) * 100, 2)
print(acc_gaussian)

# Logistic Regression
from sklearn.linear_model import LogisticRegression

logreg = LogisticRegression()
logreg.fit(x_train, y_train)
y_pred = logreg.predict(x_val)
acc_logreg = round(accuracy_score(y_pred, y_val) * 100, 2)
print(acc_logreg)

# Support Vector Machines
from sklearn.svm import SVC

svc = SVC()
svc.fit(x_train, y_train)
y_pred = svc.predict(x_val)
acc_svc = round(accuracy_score(y_pred, y_val) * 100, 2)
print(acc_svc)

# Linear SVC
from sklearn.svm import LinearSVC

linear_svc = LinearSVC()
linear_svc.fit(x_train, y_train)
y_pred = linear_svc.predict(x_val)
acc_linear_svc = round(accuracy_score(y_pred, y_val) * 100, 2)
print(acc_linear_svc)

# Perceptron
from sklearn.linear_model import Perceptron

perceptron = Perceptron()
perceptron.fit(x_train, y_train)
y_pred = perceptron.predict(x_val)
acc_perceptron = round(accuracy_score(y_pred, y_val) * 100, 2)
print(acc_perceptron)

#Decision Tree
from sklearn.tree import DecisionTreeClassifier

decisiontree = DecisionTreeClassifier()
decisiontree.fit(x_train, y_train)
y_pred = decisiontree.predict(x_val)
acc_decisiontree = round(accuracy_score(y_pred, y_val) * 100, 2)
print(acc_decisiontree)

# Random Forest
from sklearn.ensemble import RandomForestClassifier

randomforest = RandomForestClassifier()
randomforest.fit(x_train, y_train)
y_pred = randomforest.predict(x_val)
acc_randomforest = round(accuracy_score(y_pred, y_val) * 100, 2)
print(acc_randomforest)

# KNN or k-Nearest Neighbors
from sklearn.neighbors import KNeighborsClassifier

knn = KNeighborsClassifier()
knn.fit(x_train, y_train)
y_pred = knn.predict(x_val)
acc_knn = round(accuracy_score(y_pred, y_val) * 100, 2)
print(acc_knn)

# Gradient Boosting Classifier
from sklearn.ensemble import GradientBoostingClassifier

gbk = GradientBoostingClassifier()
gbk.fit(x_train, y_train)
y_pred = gbk.predict(x_val)
acc_gbk = round(accuracy_score(y_pred, y_val) * 100, 2)
print(acc_gbk)
