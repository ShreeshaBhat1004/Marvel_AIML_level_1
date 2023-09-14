# Task - 1 : Linear and Logistic Regression: The hello world of AI-ML

The first part of the task is to predict home prices based on multiple variables, using scikit's **Linear_linearRegression.model**.









# importing neccessary libraries
import pandas as pd
import numpy as np
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

# Load the California housing dataset
housing = fetch_california_housing()

# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(housing.data, housing.target, test_size=0.25)

# Create a LinearRegression model
lr = LinearRegression()

# Train the model on the training data
lr.fit(X_train, y_train)

# Make predictions on the test data
y_pred = lr.predict(X_test)

# Evaluate the model's performance on the test data
mse = mean_squared_error(y_test, y_pred)
print("MSE:", mse)


import matplotlib.pyplot as plt

# Create a scatter plot of the actual house prices vs. the predicted house prices
plt.scatter(y_test, y_pred)

# Add a line to the scatter plot that represents the prediction line
y_min = np.min(y_test)
y_max = np.max(y_test)
x_min = np.min(y_pred)
x_max = np.max(y_pred)
plt.plot([y_min, y_max], [y_min, y_max],color ='r')

# Add a title to the scatter plot
plt.title("Actual vs. Predicted House Prices")

# Add labels to the axes of the scatter plot
plt.xlabel("Actual House Prices")
plt.ylabel("Predicted House Prices")

# Show the scatter plot
plt.show()


# Linear regression Statistical analysis
- In simple linear regression we try to map relationship between 2 variables among which one is independent and another is dependent on independent variable.

            y = mx + c
            where y - dependent variable
                  x - independent variable
                  m,c - weights and biases

Observing the Variables we estimate m and c. We can also try to predict a value of y for a new value of x. If the ground truth value of y is given for the new value of x, it helps to predict the accuracy as well.

- Multiple linear regression model:
In most of the cases, the dependent variable is dependent on many variables in such cases

        y = m1x1 + m2x2 + m3x3 +...+ mkxk



  


