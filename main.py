import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv('framingham.csv')
df = df.drop('cigsPerDay',axis=1)
df = df.drop('sysBP',axis=1)
df = df.drop('education',axis=1)
df = df.drop('diaBP',axis=1)

df.head()

df.isnull().sum()

df['BPMeds'] = df['BPMeds'].fillna(df['BPMeds'].mode()[0])
df['totChol'] = df['totChol'].fillna(df['totChol'].mode()[0])
df['BMI'] = df['BMI'].fillna(df['BMI'].mode()[0])
df['heartRate'] = df['heartRate'].fillna(df['heartRate'].mode()[0])
df['glucose'] = df['glucose'].fillna(df['glucose'].mode()[0])

plt.figure(figsize=(12,6))
sns.barplot(x='age',y='TenYearCHD',data=df)

X = df.iloc[:,:-1]
X
y = df.iloc[:,-1:]
y

from sklearn.model_selection import train_test_split

X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.25,random_state= 42)

from sklearn.linear_model import LogisticRegression

classifier = LogisticRegression()

from sklearn.model_selection import GridSearchCV
parameters = {'penalty':['l1','l2','elasticnet'],'C':[1,2,4,5,8,10,30,45],'max_iter':[100,200,300]}

classifier_regressor = GridSearchCV(classifier,param_grid=parameters,scoring='accuracy',cv=10)
classifier_regressor.fit(X_train,y_train)

classifier_regressor.best_params_
y_pred = classifier_regressor.predict(X_test)

from sklearn.metrics import accuracy_score,classification_report
accuracy_score(y_pred,y_test)
print(classification_report(y_true=y_test,y_pred=y_pred))

from sklearn.metrics import confusion_matrix
confusion_matrix(y_true=y_test,y_pred= y_pred)

from imblearn.over_sampling import SMOTE

smote = SMOTE(random_state=42)
X_resampled,y_resampled = smote.fit_resample(X,y)
print(y_resampled.value_counts())

X_resampled_train,X_resampled_test,y_resampled_train,y_resampled_test = train_test_split(X_resampled,y_resampled,test_size=0.25,random_state=42)

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV

classifier = LogisticRegression()
parameters = {'penalty':['l1','l2','elasticnet'],'C':[1,2,4,5,8,10,30,45],'max_iter':[100,200,300]}
classifier_regressor = GridSearchCV(classifier,param_grid= parameters,scoring='accuracy',cv=10)
classifier_regressor.fit(X_resampled_train,y_resampled_train)

y_pred = classifier_regressor.predict(X_resampled_test)

from sklearn.metrics import classification_report,accuracy_score,confusion_matrix
accuracy_score(y_true=y_resampled_test,y_pred=y_pred)
print(classification_report(y_true= y_resampled_test,y_pred= y_pred))
confusion_matrix(y_true= y_resampled_test,y_pred= y_pred)

from sklearn.metrics import f1_score,fbeta_score
f1_score(y_true= y_resampled_test,y_pred=y_pred)


print("end of the program")