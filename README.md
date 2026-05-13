# Implementation-of-Decision-Tree-Regressor-Model-for-Predicting-the-Salary-of-the-Employee

## AIM:
To write a program to implement the Decision Tree Regressor Model for Predicting the Salary of the Employee.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm

1). Load the dataset, separate features (X) and target (y), and apply one‑hot encoding to categorical variables.

2). Initialize the DecisionTreeRegressor model with a fixed random state.

3). Train the model on the entire dataset and generate predictions.

4). Evaluate predictions (though regression needs metrics like MSE/R², not classification) and visualize the decision tree with feature names.

## Program:

Program to implement the Decision Tree Regressor Model for Predicting the Salary of the Employee.

Developed by: PORCHEZIAN S

RegisterNumber: 212225040304 

```
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor, plot_tree
from sklearn.metrics import accuracy_score, classification_report
data = pd.read_csv("Salary.csv")
X = data.drop("Salary", axis=1)
y = data["Salary"]
X = pd.get_dummies(X, drop_first=True)
#X_train, X_test, y_train, y_test = train_test_split(    X, y, test_size=0.2, random_state=42)
model = DecisionTreeRegressor(random_state=42)
model.fit(X, y)
y_pred = model.predict(X)
# Accuracy
print("\nAccuracy:", accuracy_score(y, y_pred)*100)

# Classification Report
print("\nClassification Report:")
print(classification_report(y, y_pred))

plt.figure(figsize=(25,12))
plot_tree(
    model,
    feature_names=X.columns,
    filled=True
)

plt.title("Decision Tree Regressor")
plt.show()
```
## Output:

<img width="1276" height="1078" alt="image" src="https://github.com/user-attachments/assets/318ff158-7071-4022-ab36-9cd8807e0163" />

## Result:
Thus the program to implement the Decision Tree Regressor Model for Predicting the Salary of the Employee is written and verified using python programming.
