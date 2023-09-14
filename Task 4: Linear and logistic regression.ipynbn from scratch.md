### Task 4: Linear and Logistic Regression from scratch:
#### 1. **Linear regression:**
*Understanding:* Linear regression is a statistical model which maps relationship between 2 things where one is dependent on another, based on the information it has on both things
(variables). It comes under the subclass of supervised learning.

##### **Coding linear regression from scratch**
We will use Auto insurance in sweden dataset for this task. To code from scratch we have 5 steps
- Calculate mean and variance of dataset
- calculate covariance
- calculate coefficients
- Make predictions
- predict Insurance

 Our model does not evaluate this data as number of rooms or size of house. For the model its all just numbers. This can create an unwanted bias in your machine learning model towards the columns (or features) that have higher numerical values than the others. It can also create imbalance in the variance and mathematical mean. For these reasons and also to make the job easier it is always advised to scale or normalize your features so that they all lie within the same range ( e.g. [-1 to 1] or [0 to 1] ).

### Hypothesis:
The line which passes through the data points in linear regression model is called Hypothesis

#### Coding Linear regression algorithm:
sample:


# We define some variables
import numpy as np

X = np.linspace(1,100,num = 100)*5+np.random.randn(100)*30
Y = np.linspace(1,50,num=100)

# Initialising weights,biases
W,b = np.random.randn(2)
alpha = 0.000001
W_list = [W]
b_list = [b]
J_list = []

for xi, yi in zip(X,Y):
  yhat = W*xi+b
  J = 1/2*(yhat-Y)**2
  W -= alpha*(yhat-yi)*xi
  b -= alpha*(yhat-yi)
  W_list.append(W)
  b_list.append(b)
  J_list.append(J)

plt.ylabel('House price in lakhs')
plt.xlabel('Area in square feet')
plt.plot(X,Y,'bo',markersize=3)
plt.plot(X,W*X+b)


#### Linear regression from scratch on boston dataset to predict boston house prices

# Importing libraries
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

# Load the boston housing dataset
boston_df = pd.read_csv('/content/drive/MyDrive/BostonHousing.csv')

# Splitting data into features and target
X = boston_df.drop('medv',axis = 1)
y = boston_df['medv']
# We drop medv coloumn from features and include every other coloumn
# We take medv only for our target

# Standardize the features
'''
Standardizing the features, also known as feature scaling or normalization,
is a preprocessing step that transforms the data so that all features have a
mean of 0 and a standard deviation of 1. It is a common practice in machine
learning to standardize the features before training a model.
'''
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Add a column of ones to X for the bias term
'''
Including the bias term is essential because it allows the linear regression
model to capture the intercept or baseline prediction, which is often necessary
to make accurate predictions. Without the bias term, the linear regression line
would always pass through the origin (0,0), which may not be appropriate for
many real-world datasets.
'''
X = np.c_[np.ones(X.shape[0]),X]

# split the data into training and testing sets
split_ratio = 0.8 # 80% training, 20% testing
split_idx = int(split_ratio*X.shape[0])
X_train,X_test = X[:split_idx],X[split_idx:]
y_train,y_test = y[:split_idx],y[split_idx:]


# Defining the cost function
def mse(y_true,y_pred):
  return np.mean((y_true-y_pred)**2)

# Define the function to perform linear regression
def Linear_regression(X,y,num_iterations,learning_rate):
  num_samples,num_features = X.shape
  weights = np.zeros(num_features)
  for i in range(num_iterations):
    # Calculate the predicted value
    y_pred = np.dot(X_train,weights)

    # Compute the gradients
    gradient = np.dot(X.T,y_pred-y_train)/num_samples

    # Updating the weights
    weights = weights - (learning_rate*gradient)

    # Printing the current iteration and cost
    if i%100 ==0:
      cost = mse(y_train,y_pred)
      print(f"Iteration{i+1}/{num_iterations},cost:{cost:.4f}")

  return weights

# Set the parameters
num_iterations =1000
learning_rate = 0.01

# Perform Linear regression
weights = Linear_regression(X_train,y_train,num_iterations,learning_rate)

# print the learned weights
print("Learned weights")
for feature,weight in zip(X,weights[1:]):
  print(f"{feature}: {weight:.4f}")
print(f"bias: {weights[0]:.4f}")

from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# Make predictions on the testing set
y_test_pred = np.dot(X_test, weights)

# Calculate regression metrics for testing set
test_mse = mean_squared_error(y_test, y_test_pred)
test_mae = mean_absolute_error(y_test, y_test_pred)
test_r2 = r2_score(y_test, y_test_pred)

print("\nTesting set:")
print(f"MSE: {test_mse:.4f}")
print(f"MAE: {test_mae:.4f}")
print(f"R^2: {test_r2:.4f}")

from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split

# Load the boston housing dataset
boston_df = pd.read_csv('/content/drive/MyDrive/BostonHousing.csv')

# Splitting data into features and target
X = boston_df.drop('medv',axis = 1)
y = boston_df['medv']
# We drop medv coloumn from features and include every other coloumn
# We take medv only for our target

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create a Linear Regression model
model = LinearRegression()

# Fit the model to the training data
model.fit(X_train, y_train)

# Make predictions on the testing data
y_pred = model.predict(X_test)

# Calculate evaluation metrics
mse = mean_squared_error(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

# Print the evaluation metrics
print(f"MSE: {mse:.4f}")
print(f"MAE: {mae:.4f}")
print(f"R^2: {r2:.4f}")

# Load the boston housing dataset
boston_df = pd.read_csv('/content/drive/MyDrive/BostonHousing.csv')

# Splitting data into features and target
X = boston_df.drop('medv',axis = 1)
y = boston_df['medv']
# We drop medv coloumn from features and include every other coloumn
# We take medv only for our target

import numpy as np
from sklearn.datasets import make_blobs
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split

# Calculate evaluation metrics
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

# Print the evaluation metrics
print(f"Accuracy: {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"F1 Score: {f1:.4f}")

from sklearn.datasets import make_blobs
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split

# Generate the "make_blobs" dataset
X, y = make_blobs(n_samples=1000, centers=2, random_state=42)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create a Logistic Regression model
model = LogisticRegression()

# Fit the model to the training data
model.fit(X_train, y_train)

# Make predictions on the testing data
y_pred = model.predict(X_test)

# Calculate evaluation metrics
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

# Print the evaluation metrics
print(f"Accuracy: {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"F1 Score: {f1:.4f}")
