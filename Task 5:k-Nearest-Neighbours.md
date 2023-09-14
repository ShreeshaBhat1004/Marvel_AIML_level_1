### Task 5: K-Nearest Neighbour Algorithm
The K-Nearest Neighbors (K-NN) algorithm is a non-parametric(does not assume any functional form, like y=mx + c.) machine learning algorithm used for both classification and regression tasks.

Here's how the K-NN algorithm works:

1. Input: The algorithm takes a labeled training dataset, which consists of feature vectors and their corresponding labels or values. It also takes an unlabeled test data point for which a prediction needs to be made.

1. Similarity Measure: It calculates the similarity between the test data point and each  training data point. The choice of similarity measure depends on the nature of the data. Common measures include Euclidean distance, Manhattan distance, cosine similarity, etc.

1. Finding K Neighbors: The algorithm identifies the K training data points that are closest to the test data point based on the similarity measure. K is a user-defined parameter that needs to be specified beforehand.

1. Voting or Averaging: For classification tasks, the algorithm predicts the label of the test data point by taking the majority vote among the K nearest neighbors. Each neighbor's vote is weighted according to the distance from test data point. For regression tasks, the algorithm predicts the value of the test data point by taking the average of the values of the K nearest neighbors.

1. Output: The algorithm outputs the predicted label or value for the test data point.

It's important to note some key characteristics of the K-NN algorithm:

- The value of K determines the smoothness of the decision boundary. Smaller values of K result in more complex and flexible boundaries, which can lead to overfitting. Larger values of K result in smoother boundaries but may introduce more bias. The optimal value of K depends on the dataset and problem at hand and is typically determined through experimentation or cross-validation.

- K-NN is a lazy learning algorithm, meaning it doesn't explicitly build a model during the training phase. Instead, it stores the training data and performs calculations at runtime.

- K-NN can be sensitive to the scale of the features in the dataset. It's often necessary to preprocess the data by normalizing or standardizing the features to ensure they have equal influence on the similarity calculations.

- The computational complexity of the K-NN algorithm can be high, especially for large datasets. As the number of training instances increases, the algorithm needs to calculate the similarity between the test instance and a larger number of training instances, which can be computationally expensive.

Overall, the K-NN algorithm is a simple yet powerful method for making predictions based on the similarity between data points. It is widely used in various domains such as pattern recognition, image classification, recommender systems, and anomaly detection.

#### Implementing K-Nearest Neighbours algorithm using Scikit's built in function for a classification problem
![image](https://github.com/ShreeshaBhat1004/Marvel_AIML_level_2/assets/111550331/b364e47c-2db0-44f0-9da4-5d9fbdcff69e)
Plotting accuracy vs value of knn
![image](https://github.com/ShreeshaBhat1004/Marvel_AIML_level_2/assets/111550331/4322fd9d-32cf-4592-a021-88dd1b650db9)
Hence our model will perform its best at any of the K-value between 1 and 25 except at 7 where its 
accuracy becomes slightly less
#### Implementation of K-Nearest neighbors from scratch
Steps:
1. Calculate the euclidean distance:
EUclidean distance is a way of calculating distance between two vectors of any dimension.
E.D = sqrt(sum i to N(x1_i-x2_i)^2)
2. Get nearest neighbors: Here we select few(k) nearest neighbors from the new data point by calculating the euclidean distance between them.
3. Make Predictions:
After getting k nearest neighbors for our new data point(vector), we can predict the output for that point, here in classification problem, we will predict what class does the data point belong to.

#### Implementing KNN on iris dataset:
Steps:
- Loading the dataset and converting strings to numbers
- A new function called k_nearest_neighbors function is created to implement knn, first understanding the statistics of the training data and then applying it on testing data to make predictions.
- We will use evaluate_algorithm() to evaluate the algorithm with cross-validation and accuracy_metric() to calculate the accuracy of the predictions.

```{python}
# k-nearest neighbors on the Iris Flowers Dataset
from random import seed
from random import randrange
from csv import reader
from math import sqrt

# Load a CSV file
def load_csv(filename):
    dataset = list()
    with open(filename, 'r') as file:
        csv_reader = reader(file)
        next(csv_reader)  # Skip the header row
        for row in csv_reader:
            if not row:
                continue
            dataset.append(row)
    return dataset
# Convert string column to float
def str_column_to_float(dataset, column):
	for row in dataset:
		row[column] = float(row[column].strip())

# Convert string column to integer
def str_column_to_int(dataset, column):
	class_values = [row[column] for row in dataset]
	unique = set(class_values)
	lookup = dict()
	for i, value in enumerate(unique):
		lookup[value] = i
	for row in dataset:
		row[column] = lookup[row[column]]
	return lookup

# Find the min and max values for each column
def dataset_minmax(dataset):
	minmax = list()
	for i in range(len(dataset[0])):
		col_values = [row[i] for row in dataset]
		value_min = min(col_values)
		value_max = max(col_values)
		minmax.append([value_min, value_max])
	return minmax

# Rescale dataset columns to the range 0-1
def normalize_dataset(dataset, minmax):
	for row in dataset:
		for i in range(len(row)):
			row[i] = (row[i] - minmax[i][0]) / (minmax[i][1] - minmax[i][0])

# Split a dataset into k folds
def cross_validation_split(dataset, n_folds):
	dataset_split = list()
	dataset_copy = list(dataset)
	fold_size = int(len(dataset) / n_folds)
	for _ in range(n_folds):
		fold = list()
		while len(fold) < fold_size:
			index = randrange(len(dataset_copy))
			fold.append(dataset_copy.pop(index))
		dataset_split.append(fold)
	return dataset_split

# Calculate accuracy percentage
def accuracy_metric(actual, predicted):
	correct = 0
	for i in range(len(actual)):
		if actual[i] == predicted[i]:
			correct += 1
	return correct / float(len(actual)) * 100.0

# Evaluate an algorithm using a cross validation split
def evaluate_algorithm(dataset, algorithm, n_folds, *args):
	folds = cross_validation_split(dataset, n_folds)
	scores = list()
	for fold in folds:
		train_set = list(folds)
		train_set.remove(fold)
		train_set = sum(train_set, [])
		test_set = list()
		for row in fold:
			row_copy = list(row)
			test_set.append(row_copy)
			row_copy[-1] = None
		predicted = algorithm(train_set, test_set, *args)
		actual = [row[-1] for row in fold]
		accuracy = accuracy_metric(actual, predicted)
		scores.append(accuracy)
	return scores

# Calculate the Euclidean distance between two vectors
def euclidean_distance(row1, row2):
	distance = 0.0
	for i in range(len(row1)-1):
		distance += (row1[i] - row2[i])**2
	return sqrt(distance)

# Locate the most similar neighbors
def get_neighbors(train, test_row, num_neighbors):
	distances = list()
	for train_row in train:
		dist = euclidean_distance(test_row, train_row)
		distances.append((train_row, dist))
	distances.sort(key=lambda tup: tup[1])
	neighbors = list()
	for i in range(num_neighbors):
		neighbors.append(distances[i][0])
	return neighbors

# Make a prediction with neighbors
def predict_classification(train, test_row, num_neighbors):
	neighbors = get_neighbors(train, test_row, num_neighbors)
	output_values = [row[-1] for row in neighbors]
	prediction = max(set(output_values), key=output_values.count)
	return prediction

# kNN Algorithm
def k_nearest_neighbors(train, test, num_neighbors):
	predictions = list()
	for row in test:
		output = predict_classification(train, row, num_neighbors)
		predictions.append(output)
	return(predictions)

# Test the kNN on the Iris Flowers dataset
seed(1)
filename = '/content/drive/MyDrive/Iris.csv'
dataset = load_csv(filename)
for i in range(len(dataset[0])-1):
	str_column_to_float(dataset, i)
# convert class column to integers
str_column_to_int(dataset, len(dataset[0])-1)
# evaluate algorithm
n_folds = 5
num_neighbors = 5
scores = evaluate_algorithm(dataset, k_nearest_neighbors, n_folds, num_neighbors)
print('Scores: %s' % scores)
print('Mean Accuracy: %.3f%%' % (sum(scores)/float(len(scores))))
```


