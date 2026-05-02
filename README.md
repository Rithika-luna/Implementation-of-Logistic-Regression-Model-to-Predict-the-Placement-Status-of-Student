# Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student

## AIM:
To write a program to implement the the Logistic Regression Model to Predict the Placement Status of Student.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
```
1. Import the required libraries and load the California Housing dataset.
2.Select the input features and target variables, then split the dataset into training and testing data.
3.Normalize the training and testing data using StandardScaler.
4.Create and train the SGD Regressor model using MultiOutputRegressor.
5.Predict the house price and number of occupants, then evaluate and display the results.
```
## Program:
```
/*
Program to implement the the Logistic Regression Model to Predict the Placement Status of Student.
Developed by: S Rithika
RegisterNumber: 212225040344

import pandas as pd
import numpy as np
df = pd.read_csv('Placement_Data.csv')
df
df1 = df.copy()
df1
df1 = df1.drop(['sl_no', 'salary'], axis=1)
df1.isnull().sum()
df1.duplicated().sum()
df1
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
df1['gender'] = le.fit_transform(df1['gender'])
df1['ssc_b'] = le.fit_transform(df1['ssc_b'])
df1['hsc_b'] = le.fit_transform(df1['hsc_b'])
df1['hsc_s'] = le.fit_transform(df1['hsc_s'])
df1['degree_t'] = le.fit_transform(df1['degree_t'])
df1['workex'] = le.fit_transform(df1['workex'])
df1['specialisation'] = le.fit_transform(df1['specialisation'])
df1['status'] = le.fit_transform(df1['status'])
df1
x = df1.iloc[:, :-1]
y = df1['status']
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0)
from sklearn.linear_model import LogisticRegression
model = LogisticRegression(solver="liblinear")
model.fit(x_train, y_train)
y_pred = model.predict(x_test)
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
accuracy = accuracy_score(y_test, y_pred)
confusion = confusion_matrix(y_test, y_pred)
cr = classification_report(y_test, y_pred)
print("Accuracy Score:", accuracy)
print("\nConfusion Matrix:\n", confusion)
print("\nClassification Report:\n", cr)
from sklearn import metrics
cn_display = metrics.ConfusionMatrixDisplay(confusion_matrix=confusion, display_labels=['true', 'false'])
cn_display.plot()

*/
```

## Output:

<img width="706" height="707" alt="image" src="https://github.com/user-attachments/assets/1b12e6ca-5b3e-4215-8627-8c5fdd396b31" />


## Result:
Thus the program to implement the the Logistic Regression Model to Predict the Placement Status of Student is written and verified using python programming.
