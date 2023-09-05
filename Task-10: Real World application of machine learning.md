# Task 10: Exploration of a real world application of machine learning
### What's currently in trend in machine learning application: 
- **Fraud Detection**: Machine learning models can be used to identify patterns and anomalies in financial transactions, helping detect fraudulent activities and protect against credit card fraud, insurance fraud, and other types of fraudulent behavior.

- **Image Recognition**: Machine learning algorithms can be trained to recognize and classify objects in images. This technology is used in a wide range of applications, including facial recognition, autonomous vehicles, and medical imaging.

- **Natural Language Processing (NLP)**: NLP techniques enable machines to understand and process human language. Applications include virtual assistants, chatbots, sentiment analysis, and language translation.

- **Recommendation Systems**: Machine learning is used to build recommendation systems that analyze user preferences and behaviors to provide personalized recommendations. Examples include movie recommendations on streaming platforms like Netflix and product recommendations on e-commerce websites like Amazon.

- **Autonomous Vehicles**: Machine learning plays a crucial role in developing self-driving cars. It involves training models to recognize and interpret data from sensors, such as cameras and LiDAR, to make decisions related to steering, acceleration, and braking.

- **Healthcare Diagnostics**: Machine learning models can assist in medical diagnostics by analyzing patient data, such as medical images, lab results, and electronic health records. This can help in early detection of diseases, predicting patient outcomes, and guiding treatment decisions.

- **Customer Churn Prediction**: By analyzing historical customer data, machine learning models can predict which customers are likely to churn (cancel their subscription or stop using a service). This allows companies to take proactive measures to retain those customers.

- **Sentiment Analysis**: Sentiment analysis uses machine learning to analyze and classify opinions expressed in text data, such as social media posts, customer reviews, and survey responses. It helps businesses understand public opinion, monitor brand reputation, and make data-driven decisions.

- **Energy Demand Forecasting**: Machine learning models can analyze historical energy consumption data, weather patterns, and other relevant factors to forecast future energy demand. This is crucial for optimizing energy generation, distribution, and pricing.

- **Financial Market Prediction**: Machine learning algorithms can analyze historical financial data and market indicators to predict stock prices, currency exchange rates, and other market trends. These predictions are used by traders, investors, and financial institutions for decision-making.
### Case study: Recommendation systems:
**General**
Recommendation systems or techniques are algorithms trained to recommend preferable content or item for the customer based on their previous transactions.Recommendation systems are widely used such as in E-Commerce,Streaming sevices,Social media etc.
**How is machine learning used here**
 some commonly used machine learning algorithms in recommendation systems:
### Collaborative Filtering:
**Collaborative filtering** 
Collaborative Filtering Systems predict what you like depending on what other similar users have liked in the past.


In this approach, the algorithm follows the following approach step by step:

1. Consider user X

2. Find set N (Neighborhood of user X) of other users whose ratings are “most similar” to X’s ratings

3. Estimate X’s ratings based on ratings of users in N

Measuring the “most similar”
Let us consider a set of four users (U1, U2, U3, U4) as rows of a table and a set of movies (M1, M2…M7) as the columns of the table. Imagine a scale of ratings from 0 to 5. We have some of the ratings given by the users for the corresponding movies in the cells, while some of the cells are empty or has missing values.
![](https://miro.medium.com/v2/resize:fit:828/format:webp/1*d7umJVVWgNuFXp6owipDVw.png)

If we look at the above table carefully, we will notice that U1 and U2 have rated only one movie (M1) in common but the ratings are fairly high. This implies that users U1 andU2 have similar tastes or interests. Whereas although the users U1 and U3 have rated two movies in common, the ratings are fairly low. Thus, we can conclude that intuitively that the users U1 and U3 are dissimilar and U1 and U2 are similar.

SIM (U1, U2) > SIM (U1, U3)

To calculate the similarity matrix between the two points, U1 and U2, we use the Cosine Similarity:

SIM (U1, U2) = cosine (RU1, RU2),

Where RU1, RU2 are the rating vectors of U1 and U2 respectively.

But in order to implement the cosine similarity, we need to calculate and fill in the empty values. So, instead of a cosine similarity, we need to normalize the ratings by subtracting the row mean and use what is known as the ‘Centered Cosine Similarity’.

Below is the table we get after normalizing the ratings:
![](https://miro.medium.com/v2/resize:fit:1100/format:webp/1*UaMzVFRbuLfj8GVbwaUgPA.png)

If you observe, you will notice that the addition of all the ratings for a particular user yields to zero. This is because the ratings have now been centered around zero. The ratings above zero show a positive or high rating and those below zero indicate negative or low rating. Now, calculating the ‘Centered Cosine Similarity’ between users U1, U2 and U1, U3, we get:

SIM (U1, U2) = cosine (RU1, RU2) = 0.09, and

SIM (U1, U3) = cosine (RU1, RU3) = -0.56

SIM (U1, U2) > SIM (U1, U3)

The results indicate that U1 and U2 are highly similar whereas U1 and U3 are very unlike each other.

Now that we have estimated and grouped similar and dissimilar users, the next step is to make rating predictions for a user.

Rating Predictions:
Suppose a user x has a rating vector Rx, and we are required to make rating predictions for this user for item i. Using Centered Cosine Similarity, we find a neighborhood N of a set of k users (who have rated item i) who are the most similar to user x. To do so, we take the weighted average. In this method, for each user y in neighborhood N, we weight y rating for item i and multiply it by the similarity of x and y. Finally, we normalize it by dividing the product by the sum of the similarities between x and y. The result gives us an estimate for user x and item y.
![](https://miro.medium.com/v2/resize:fit:640/format:webp/1*VKSZ_nD1ScbAafWDzF2PwA.png)

where Sxy = SIM (x, y)

The technique that we used above is also sometimes called as the User-User Collaborative Filtering, as we try to find other users with similar interests in order to make predictions for other similar users. A dual approach to Collaborative Filtering is the Item-Item Collaborative Filtering, where instead of starting out with a user, we start with an item i and find a set of other items similar to i. Then we estimate rating for item i based on the ratings for similar items. We can use the same similar metrics and prediction approaches as the User-User model.

Let us consider a group of users U1, U2… U12 as the columns of the table and a set of movies M1, M2… M6 as the rows of the table.
![](https://miro.medium.com/v2/resize:fit:828/format:webp/1*1OY58dLFXb03pzaiXVz87Q.png)

The yellow cells have known ratings (in the scale of 1 to 5) and the white ones have empty ratings. And our goal here is to find the rating for movie 1 (M1) by user 5 (U5). The first step would be to find other movies that are similar to M1. To calculate similarity, we use the Pearson Correlation technique, which is analogous to the Centered Cosine technique, we used earlier. So, we take all the movies and calculate their individual centered cosine distances and list them up, with respect to M1.
![](https://miro.medium.com/v2/resize:fit:828/format:webp/1*xFmCReh9bfuybmAKAFEeLg.png)

If we take into account the movies that have positive centered cosine similarity with M1, we have two neighbors in the neighborhood N i.e. N=2. Using the same methods as in the case of User-User Collaborative Filtering, we calculate the similarity between M1 and M3, and between M1 and M6. We get:

SIM (M1, M3) = 0.14, and SIM (M1, M6) = 0.59

Taking the weighted average (using the formula mentioned earlier), we find that the predicted rating for user U1 for movie M1 is 2.6.


**Content-Based Filtering**: 
Content based Recommender System:
Content based Recommender Systems predict what you like depending on what you’ve liked in the past.


The diagram above shows a basic plan of action for the content based recommender system. We have it explained in the points below:

- Find a set of items liked by the user, by both explicit and implicit methods. For example, the items purchased by the user

- Using those set of items, an Item Profile is built — which is essentially a description of the items purchased by the user. Here in the diagram, the geometric shapes have been used for the sake of succinctness. So, we can conclude that the user likes items that are red in color, and are in the shapes of circles and triangles

- Next, from the item profile, we are going to infer a User Profile, that would contain information about the user regarding his/her likes and purchases

Now that we have the user and the item profiles, the next task would be to recommend certain items to the user.

- Given the user, we compute the similarity between that user and all the items available in the catalog. The similarity is calculated using the Cosine Similarity technique.

- We then pick the item with the highest cosine similarity, and recommend those to the user.
# Collaborative filtering v/s Content based filtering approach:
|Collaborative Filtering                  |Content based filtering                         |
|------------------------------------------------------------------------------------------|
|Content-based filtering methods require  |Collaborative filtering, on the other hand, uses| 
quite an amount of information about an   |historical interactions between the users and   |
item’s features, rather than its          |and items to group users with similar tastes and|
interactions with the user. For products  |suggest new items, which are popular to the     |
like clothes, these features can be size, | to the target user.                            |
color, brand, material, etc., or in the   |                                                |
case of movies, actors, genre, director,  |                                                |
year of release, etc.                     |                                                |
--------------------------------------------------------------------------------------------
**Conclusion**
In conclusion, recommendation systems have become an integral part of our digital lives, revolutionizing the way we discover content, products, and services. These systems leverage machine learning algorithms and techniques to analyze user behavior, preferences, and historical data, enabling them to provide personalized and relevant recommendations. From e-commerce and streaming platforms to social media and travel websites, recommendation systems have found applications in various domains, enhancing user experiences, boosting engagement, and driving business growth. With ongoing advancements in machine learning and data analytics, we can expect recommendation systems to continue evolving, delivering even more accurate and tailored recommendations. As users, we can look forward to a future where our interactions with technology are seamlessly personalized, making our digital journeys more enjoyable and efficient than ever before.

