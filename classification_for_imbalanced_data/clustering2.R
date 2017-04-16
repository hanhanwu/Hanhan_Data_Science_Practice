# elbow method plot - find optimal 

mydata <- scaled_data
wss <- (nrow(mydata)-1)*sum(apply(mydata,2,var))
for (i in 2:20) wss[i] <- sum(kmeans(mydata,centers=i)$withinss)
plot(1:20, wss, type="b", xlab="Number of Clusters", ylab="Within groups sum of squares")

# silhouette coefficient, check cluster similarity, higher the better
library (cluster)
library (vegan)
dis = vegdist(mydata)
res = pam(dis,10)
sil = silhouette(res$clustering,dis) # or use your cluster vector
plot(sil, border = NA)

# hierarchical cluster
d <- dist(scaled_data, method = "euclidean") # Euclidean distance matrix.
H.fit <- hclust(d, method="ward.D")   # Ward’s minimum variance criterion minimizes the total within-cluster variance
par(mar=rep(2,4))
plot(H.fit) # display dendogram
groups <- cutree(H.fit, k=5) # cut tree into 5 clusters
# draw dendogram with red borders around the 5 clusters
rect.hclust(H.fit, k=5, border="red") 

# Plot Y, N distribution in clusters
k.means.fit <- kmeans(normalized_data, 2)
k.means.fit$centers
k.means.fit$cluster
clust_label <- cbind(k.means.fit$cluster, data_label)
colnames(clust_label)[1] <- "Cluster"
colnames(clust_label)[2] <- "IsUnderrated"
clust_label <- data.frame(clust_label)
head(clust_label)
par(mfrow=c(2,1), mai=c(0.5, 0.5, 0.2, 0.2))    # mai here is to create lower chart
Y <- table(clust_label$Cluster[which(clust_label$IsUnderrated=='Y')])
barplot(Y, main="Y")
N <- table(clust_label$Cluster[which(clust_label$IsUnderrated=='N')])
barplot(N, main="N")


# Cluster Ensembling
library(clue)
d <- dist(scaled_data, method = "euclidean") # Euclidean distance matrix.
hclust_methods <- c("ward.D", "single", "complete", "average", "mcquitty")
hclust_results <- lapply(hclust_methods, function(m) hclust(d, m))
names(hclust_results) <- hclust_methods
## Now create an ensemble from the results.
hens <- cl_ensemble(list = hclust_results)
hens
## Subscripting.
hens[1 : 3]
## Replication.
rep(hens, 3)
## Plotting.
plot(hens, main = names(hens))
## And continue to analyze the ensemble, e.g.
round(cl_dissimilarity(hens, method = "gamma"), 4)


# Clustering Y group and N group
scaled_data$IsUnderrated <- data_label
Y_group <- subset(scaled_data, IsUnderrated=="Y")
N_group <- subset(scaled_data, IsUnderrated=="N")

Y_group[, IsUnderrated:=NULL]
mydata <- Y_group
wss <- (nrow(mydata)-1)*sum(apply(mydata,2,var))
for (i in 2:20) wss[i] <- sum(kmeans(mydata,centers=i)$withinss)
plot(1:20, wss, type="b", xlab="Number of Clusters", ylab="Within groups sum of squares")

d <- dist(Y_group, method = "euclidean") # Euclidean distance matrix.
hclust_methods <- c("ward.D", "single", "complete", "average", "mcquitty")
hclust_results <- lapply(hclust_methods, function(m) hclust(d, m))
names(hclust_results) <- hclust_methods
## Now create an ensemble from the results.
hens <- cl_ensemble(list = hclust_results)
hens
## Subscripting.
hens[1 : 3]
## Replication.
rep(hens, 3)
## Plotting.
plot(hens, main = names(hens))
