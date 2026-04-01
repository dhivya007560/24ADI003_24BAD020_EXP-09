# 24ADI003_24BAD020_EXP-09
DATASET LINK:https://www.kaggle.com/datasets/rajmehra03/movielens100k

SCENARIO 1 – USER-BASED COLLABORATIVE FILTERING:
Problem Statement: Recommend movies to users based on similar users' preferences.

Description:
            This project implements a User-Based Collaborative Filtering Recommendation System using the MovieLens dataset. The system recommends movies to users by identifying other users with similar preferences and leveraging their ratings.
 Methodology:
            Constructed a User-Item Matrix from user ratings.
            Applied mean-centering to normalize user rating behavior.
            Computed user similarity using Cosine Similarity.
            Selected top-K similar users (neighbors).
            Predicted ratings using weighted average of neighbor ratings.
            Recommended top-N unseen movies for each user.
 Features Used:
      User ID
      Movie ID
      Ratings
 Evaluation Metrics:
      Root Mean Square Error (RMSE)
      Mean Absolute Error (MAE)
 Output:
       Top recommended movies for a given user.
       Predicted ratings.
       User similarity matrix.
       Visualization graphs (heatmaps and bar charts).
 Advantages:
      Provides personalized recommendations.
      Easy to implement and interpret.
 Limitations:
      Suffers from data sparsity.
      Not scalable for large datasets.
      Cold-start problem for new users.

 
 SCENARIO 2 – ITEM-BASED COLLABORATIVE FILTERING:
 Problem Statement:
Recommend similar items (movies/products) based on user ratings.
 Description:
          This project implements an Item-Based Collaborative Filtering Recommendation System where similar items (movies) are recommended based on user rating patterns. The system identifies relationships between items instead of users.

Methodology:
      Created an Item-User Matrix.
      Applied mean-centering for normalization.
      Computed similarity between items using Cosine Similarity.
      Identified top similar items for each movie.
      Generated recommendations based on user history.
      Ranked items using weighted similarity scores.

Features Used:
      Item (Movie) ID.
      User ratings.
      User interactions.
      
Evaluation Metrics:
      Root Mean Square Error (RMSE)
      Precision@K

Output:
      Top similar items.
      Recommended movies for users.
      Item similarity matrix.
      
Visualization graphs:
      Heatmap of item similarity.
      Top similar items graph.
      Recommendation bar chart.

Advantages:
       More scalable than user-based filtering.
       Stable recommendations over time.
       Works well for large datasets.

Limitations:
       Requires sufficient item interactions.
       Less personalized compared to user-based filtering.
