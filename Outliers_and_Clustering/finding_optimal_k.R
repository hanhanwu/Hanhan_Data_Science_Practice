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

# PAM clustering, Partitioning Around Medoids (PAM), a type of k-modoids method
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
# Method 1.2 - Elbow Method DIY
set.seed(410)
max_k <- 15
data <- iris_scaled
## total within-cluster sum of squares for each k
wss <- sapply(1:max_k, function(k){kmeans(data, k, nstart = 10)$tot.withinss})
wss
plot(1:max_k, wss, type = "b", pch=19, frame=F, xlab = "Number of Clusters",
     ylab = "Total within-clusters summ of squares")
abline(v=3,lty=2)  # use the line indicates the best k, you tell which k is the best through v

# Method 1.2 - Elbow Method Built-in (package factorextra): kmeans
fviz_nbclust(iris_scaled, kmeans, method = "wss") + geom_vline(xintercept = 3, linetype=2)
# Method 1.3 - Elbow Method Built-in (package factorextra): PAM
fviz_nbclust(iris_scaled, pam, method = "wss") + geom_vline(xintercept = 3, linetype=2)
# Method 1.4 - Elbow Method Built-in (package factorextra): hierarchical clustering
fviz_nbclust(iris_scaled, hcut, method = "wss") + geom_vline(xintercept = 3, linetype = 2)


# Method 2.1 - Average Silhouette Score DIY
max_k <- 15
data <- iris_scaled
ss<- sapply(2:max_k, function(k){mean(silhouette(kmeans(data, k, nstart = 10)$cluster,dist(data))[, 3])})
ss
sil = c(0, ss)
sil
plot(1:max_k, sil, type = "b", pch=19, frame=F, xlab = "Number of Clusters",
     ylab = "Silhouette Score")
abline(v=2,lty=2)

# Method 2.2 - Average Silhouette Score Built-in (package factorextra): kmeans
fviz_nbclust(iris_scaled, kmeans, method = "silhouette")
# Method 2.3 - Average Silhouette Score Built-in (package factorextra): PAM
fviz_nbclust(iris_scaled, pam, method = "silhouette")
# Method 2.3 - Average Silhouette Score Built-in (package factorextra): hierarchical clustering
fviz_nbclust(iris_scaled, hcut, method = "silhouette", hc_method="complete")


# Method 3.1 - Gap Statistics DIY
set.seed(410)
gap_stat <- clusGap(iris_scaled, FUN=kmeans, nstart=25, K.max=10, B=50)  # B means bootstrap
print(gap_stat, method = "firstmax")
plot(gap_stat, frame=F, xlab = "Number of Clusters")
abline(v=3,lty=2)
# Method 3.2 - Gap Statistics Built-in (package factorextra): kmeans
gap_stat <- clusGap(iris_scaled, FUN=kmeans, nstart=10, K.max=10, B=50)
fviz_gap_stat(gap_stat)
# Method 3.3 - Gap Statistics Built-in (package factorextra): PAM
gap_stat <- clusGap(iris_scaled, FUN=pam, K.max=10, B=50)
fviz_gap_stat(gap_stat)
# Method 3.4 - Gap Statistics Built-in (package factorextra): Hierarchical Clustering
gap_stat <- clusGap(iris_scaled, FUN=hcut, K.max=10, B=50)
fviz_gap_stat(gap_stat)


# Method 4 - With NbClust, you can use 30 algorithms (index) to calculate optimal k 
## and get the k with the highest votes
set.seed(410)
nb <- NbClust(iris_scaled, distance = "euclidean", min.nc = 2, max.nc = 15,
              method = "complete", index = "silhouette")
nb  # try silhouette first

nb <- NbClust(iris_scaled, distance = "euclidean", min.nc = 2, max.nc = 15,
              method = "complete", index = "all")
nb  # try all the 30 algorithms
fviz_nbclust(nb) + theme_minimal()  # get votes for k
