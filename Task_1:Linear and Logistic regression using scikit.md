The first part of the task is to predict home prices based on multiple variables, using scikit's Linear_linearRegression.model.
![image](https://github.com/ShreeshaBhat1004/Marvel_AIML_level_2/assets/111550331/14eb33e8-fd8a-4800-b360-161c79ac50e2)
We get a MSE of 0.5451 which is less error
![image](https://github.com/ShreeshaBhat1004/Marvel_AIML_level_2/assets/111550331/9bb5dacd-31cb-40e6-bcbf-e0351933a61a)
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

# Logistic regression:
Logistic regression is used in predicting the outcome of binary possibilities such as win or lose, yes or no.

Now the task is to train a model to distinguish between different species of the iris flower based on sepal length, sepal width, petal length and petal width.

So, we have the independent variables(x) as sepal length, sepal width, petal length and petal width. And the label is different species of iris flower.
![image](https://github.com/ShreeshaBhat1004/Marvel_AIML_level_2/assets/111550331/21553574-1a4b-48a3-b62b-148530fd99a5)
*As you can see from the image, the three species of the iris which we are going to classify using sepal,petal - length and width, using scikit.learn_model's LogisticRegression model*

This includes few steps:
- import required libraries
- load the dataset
- split the dataset into training and test dataset
- Train the model based on training data
- make predictions
- Check accuracy

Importing libraries
![image](https://github.com/ShreeshaBhat1004/Marvel_AIML_level_2/assets/111550331/887db3fe-3da1-4a84-8ca7-45374d0fea40)
Loading the dataset
![image](https://github.com/ShreeshaBhat1004/Marvel_AIML_level_2/assets/111550331/b55590d3-d46e-4e09-839e-abbd35659d21)
Splitting the dataset
![image](https://github.com/ShreeshaBhat1004/Marvel_AIML_level_2/assets/111550331/487da986-5632-465e-b74a-496c2716f791)
Fitting the training data into the model
![image](https://github.com/ShreeshaBhat1004/Marvel_AIML_level_2/assets/111550331/8bdae027-0dc9-4ae9-9511-842f92ecfdb8)
Checking the accuracy
![image](https://github.com/ShreeshaBhat1004/Marvel_AIML_level_2/assets/111550331/7968bf2e-ae1a-4064-8664-13bd5b2174d7)

Thus we get a 100% accuracy for our model

