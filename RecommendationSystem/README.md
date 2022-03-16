
## MAJOR METHODS

### Content Based Filtering
* This method tend to introduce same category products. So if a user didn't click/like a certain category, that category products won't be recommended.
* Content based filtering tend to use <b>similarity based method</b> such as cosin similarity, <b>distance based method</b> such as euclidean distance, or <b>correlation based method</b> such as pearson correlation to make similar recommendations.

### Collaborative Filtering
* Collaborative Filtering is possible to recommends products in other categories that a user has never tried before.
* User-User Recommendation
  * The basic idea here is, compare a user with all the other users, based on their ratings. Recommend products highly rated by similar users.
  * The formula is using `common items rated by the 2 users`
  * To compare a user with all the other users can be very time consuming. We can narrow down the search space in these methods:
    * Clustering, and use users in the same cluster
    * Choose users tend to be similar, such as friends, colleagues, etc. (depends on the situation)
    * Random selection
  * Or, we can try Item-Item collaborative filtering
* Item-Item Recommendation
  * Recommend similar items
  * The formula is using `common users who rate the 2 items`
* Cold Start
  * When there is new user or new item added, no history
    * Visitor cold start
    * Product cold start
  * While collaborative filtering needs a certain amount of history, content based filtering can help deal with cold start

### Ranking Methods
* [LGBM Ranker][6]
  * [A simple example][7] 
  * [FLAML also provides param tuning method for this][8]
  
### Words Embedding
* It makes recommendations based on the similarity of words vectors
  * Words in similar info context tend to have similar vectors
* [Example - Recommendation System with Word2Vec][1]
  
### Evaluation Metrics
* Positive means the user likes the recommended item
* RMSE
  * How accurate the recommendation is
  * Do NOT consider c
* Precision & Recall
  * `precision = TP/(TP+FP)`, it measures for the recommended items, the propotion that the user really like
  * `recall = TP/(TP+FN)`, for the items that a user really likes, the propotion that got recommended
  * Do NOT consider recommendation orders
* Mean Reciprocal Rank
   * Considers recommendation order
* MAP at k (Mean Average Precision at cutoff k)
  * By calculating cutoff precision, it will depends on the order. Fianlly get the averaged value
  * Considers recommendation order
* NDCG (Normalized Discounted Cumulative Gain)
  * MAP uses interest or not (binary), NDCG uses a score
  * Considers recommendation order
  
  
## PRACTICE
* MovieLens Recommendation System
  * My code: https://github.com/hanhanwu/Hanhan_Data_Science_Practice/blob/master/RecommendationSystem/recommendation_system_movielens.ipynb
    * Download the data from : https://grouplens.org/datasets/movielens/100k/
    * Method 1 - DIY collaborative filtering
    * Method 2 - Library `turicreate` for recommendations
      * This one really gave me a hard time. Not sure why, I could not use `SFrame()` and found a solution to install `graphlab`, which could only be installed in my conda virtual environment and broke my ipython kernel, I had to uninstall and reinstall IPython and kernel.... Then it still not work, but later worked...
        * To install turicreate: `pip install -U turicreate`
        * To install graphlab: https://turi.com/download/install-graphlab-create-command-line.html
        * To reinstall IPython and kernel
          * `pip uninstall ipython`
          * `pip install ipython`
          * `pip install ipykernel`
          * `python -m ipykernel install --user --name testenv --display-name "Python2 (yourenvname)"`
          * `conda install ipywidgets --no-deps`
      * To check models in turicreate: https://apple.github.io/turicreate/docs/api/turicreate.toolkits.html
        * `recommender` models in turicreate: https://apple.github.io/turicreate/docs/api/turicreate.toolkits.recommender.html#creating-a-recommender
          * Pretty cool, it has not only collaborative filtering, but also content based recommendation, factorization recommendation
          * parameters for the collaboritive filtering recommender: https://apple.github.io/turicreate/docs/api/generated/turicreate.recommender.item_similarity_recommender.create.html#turicreate.recommender.item_similarity_recommender.create
            * In the code it's `cosin` as similarity type, we can also use `pearson`, `jaccard`
      * <b>3 types of recommenders in the code</b>
        * Popularity Recommender - it recommends most popular items to everyone, exactly the same items and orders
        * Collaboritive Filtering - recommend items based on ratings
          * It can recommend similiar products based on different user ratings and your target user's ratings
          * It can also recommend similar users based on same items rated by different users, just transpose the matrix above, which will reverse user and item
        * Rating Predition - Predict missing ratings, since each user may not rate all the items
          * When using this method, you have to use `user_id`, `item_id` as colum names, otherwise turicreate will return error. This is what I don't really like
          * The idea is the same as matrix factorization, and it also uses gradient descent to do optimization
  * Reference: https://www.analyticsvidhya.com/blog/2018/06/comprehensive-guide-recommendation-engine-python/?utm_source=feedburner&utm_medium=email&utm_campaign=Feed%3A+AnalyticsVidhya+%28Analytics+Vidhya%29
    * The description is good. It implemented its own matrix facotrization, I used turicreate built-in
    
* Recommendation with Linear Optimization
  * [YouTube video selection][4]
  * Minimize both criteria in this case
* [Ranking with weighted criteria][5]
  * The better solution here is, get the relative values by diving the max value in each criteria

## My Applied Projects
* [Golden Bridge - bridge individual & business users][2]
* [Automatic & Interactive FLinraud Detection][3]

## Applications
* [Recommendation engine with simple algorithms][9]

[1]:https://www.analyticsvidhya.com/blog/2019/07/how-to-build-recommendation-system-word2vec-python/?utm_source=blog&utm_medium=graph-feature-extraction-deepwalk
[2]:https://github.com/hanhanwu/Hanhan_Break_the_Limits/tree/master/Bank_Fantasy/Golden_Bridge
[3]:https://github.com/hanhanwu/Hanhan_Break_the_Limits/tree/master/attack_signals_recommendation_system
[4]:https://github.com/hanhanwu/Hanhan_Play_With_Social_Media/blob/master/DEF_CON_video_list_linear_optimization.ipynb
[5]:https://www.analyticsvidhya.com/blog/2020/09/how-to-rank-entities-with-multi-criteria-decision-making-methodsmcdm/?utm_source=feedburner&utm_medium=email&utm_campaign=Feed%3A+AnalyticsVidhya+%28Analytics+Vidhya%29
[6]:https://lightgbm.readthedocs.io/en/latest/pythonapi/lightgbm.LGBMRanker.html
[7]:https://stackoverflow.com/questions/64294962/how-to-implement-learning-to-rank-using-lightgbm/67627169#67627169
[8]:https://github.com/microsoft/FLAML#examples
[9]:https://www.analyticsvidhya.com/blog/2022/03/a-comprehensive-guide-on-recommendation-engines-and-implementation/?utm_source=feedburner&utm_medium=email
