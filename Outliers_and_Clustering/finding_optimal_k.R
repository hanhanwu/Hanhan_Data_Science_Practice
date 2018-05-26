# Methods to find optimal k for clustering
# Clustering using k-mean, PAM and hierarchical clustering
library(factoextra)
library(cluster)
library(NbClust)

# data preparation
data(iris)
head(iris)
iris_scaled <- scale(iris[,-5])
head(iris_scaled)

set.seed(410)

# k-means
km <- kmeans(iris_scaled, 3, nstart = 25)  # 25 random sets should be chosen
km$cluster
## visualize clusters
fviz_cluster(km, data = iris_scaled, geom = "point", stand = F, frame.type = "norm")

# PAM clustering
pam <- pam(iris_scaled, 3)
pam$cluster
fviz_cluster(pam, stand = F, geom = "point", frame.type = "norm")

# Hierarchical Clustering
clust_dist <- dist(iris_scaled, method = "euclidean")
hc <- hclust(clust_dist, method = "complete")
plot(hc, labels = F, hang = -1)  # hang=-1 so that the bottom looks tidy
rect.hclust(hc, k=3, border = 2:4)  # add rectangle to each cluster, with k colors
## cut into 3 groups (similar to km$cluster, pam$cluster above)
hc.cut <- cutree(hc, k=3)
hc.cut

############################# Methods to Determine Optimal k #############################
