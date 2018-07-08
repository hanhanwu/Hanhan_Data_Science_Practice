
*****************************************************************************

MAJOR METHODS

1. <b>Content Based Filtering</b>
* This method tend to introduce same category products. So if a user didn't click/like a certain category, that category products won't be recommended.
* Content based filtering tend to use <b>similarity based method</b> such as cosin similarity, <b>distance based method</b> such as euclidean distance, or <b>correlation based method</b> such as pearson correlation to make similar recommendations.
2. <b>Collaborative Filtering</b>
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
  
Practice

* MovieLens Recommendation System
  * My code: https://github.com/hanhanwu/Hanhan_Data_Science_Practice/blob/master/RecommendationSystem/recommendation_system_movielens.ipynb
    * Method 1 - DIY collaborative filtering
    * Method 2 - Library `turicreate` for collaborative filtering
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
