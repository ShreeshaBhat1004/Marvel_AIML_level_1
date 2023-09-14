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

