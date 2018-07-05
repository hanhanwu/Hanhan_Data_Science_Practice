
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
