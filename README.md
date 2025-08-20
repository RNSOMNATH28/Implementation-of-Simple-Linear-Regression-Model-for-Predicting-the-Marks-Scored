# Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored

## AIM:
To write a program to predict the marks scored by a student using the simple linear regression model.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. import the needed packages.
2. Assigning hours to x and scores to y.
3. Plot the scatter plot.
4. Use mse,rmse,mae formula to find the values.

## Program:
```
/*
Program to implement the simple linear regression model for predicting the marks scored.
Developed by: R N SOMNATH
RegisterNumber: 212224240158 
*/
```
```
# IMPORT REQUIRED PACKAGE
import pandas as pd
import numpy as np
from sklearn.metrics import mean_absolute_error,mean_squared_error
import matplotlib.pyplot as plt
dataset=pd.read_csv('student_scores.csv')
print(dataset)
# READ CSV FILES
dataset=pd.read_csv('student_scores.csv')
print(dataset.head())
print(dataset.tail())
# COMPARE DATASET
x=dataset.iloc[:,:-1].values
print(x)
y=dataset.iloc[:,1].values
print(y)
# PRINT PREDICTED VALUE
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=1/3,random_state=0)
from sklearn.linear_model import LinearRegression
reg=LinearRegression()
reg.fit(x_train,y_train)
y_pred = reg.predict(x_test)
print(y_pred)
print(y_test)
# GRAPH PLOT FOR TRAINING SET
plt.scatter(x_train,y_train,color='purple')
plt.plot(x_train,reg.predict(x_train),color='black')
plt.title("Hours vs Scores(Training set)")
plt.xlabel("Hours")
plt.ylabel("Scores")
plt.show()
# GRAPH PLOT FOR TESTING SET
plt.scatter(x_test,y_test,color='red')
plt.plot(x_train,reg.predict(x_train),color='black')
plt.title("Hours vs Scores(Testing set)")
plt.xlabel("Hours")
plt.ylabel("Scores")
plt.show()
# PRINT THE ERROR
mse=mean_absolute_error(y_test,y_pred)
print('Mean Square Error = ',mse)
mae=mean_absolute_error(y_test,y_pred)
print('Mean Absolute Error = ',mae)
rmse=np.sqrt(mse)
print("Root Mean Square Error = ",rmse)

```

## Output:
   To Read Head and Tail Files
<img width="192" height="130" alt="image" src="https://github.com/user-attachments/assets/d2888782-4aad-4307-8162-9699d2b5daac" />
<img width="631" height="487" alt="image" src="https://github.com/user-attachments/assets/f8a8d9a3-564a-4255-87ca-0efe146fc23a" />
  Compare Dataset
<img width="756" height="72" alt="image" src="https://github.com/user-attachments/assets/e8298586-a7d5-42ab-991f-b8c42d7bc5b4" />
  Predicted Value
<img width="793" height="566" alt="image" src="https://github.com/user-attachments/assets/471bcdd7-45cb-4f8f-9e25-b478ba731206" />
  Graph For Testing Set
<img width="803" height="576" alt="image" src="https://github.com/user-attachments/assets/fd89d8a3-01a1-48a9-a189-140119c83be3" />
  Error
<img width="450" height="71" alt="image" src="https://github.com/user-attachments/assets/c8c20a14-c9ec-4fbc-80a3-1809e07328c0" />
## Result:
Thus the program to implement the simple linear regression model for predicting the marks scored is written and verified using python programming.
