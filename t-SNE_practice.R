library(Rtsne)
library(mlr)

path<- "[your path of the data file]"
setwd(path)

train_data <- read.csv("train.csv")
summarizeColumns(train_data)     # 100 features, not that much...

labels <- train_data$label
colors <- rainbow(length(unique(train_data$label)))   
names(colors) <- unique(train_data$label)
tsne <- Rtsne(train_data[, -1], dims = 2, perplexity=30, verbose=TRUE, max_iter = 500)   # 51.88 sec
exeTimeTsne<- system.time(Rtsne(train_data[, -1], dims = 2, perplexity=30, verbose=TRUE, max_iter = 500))

plot(tsne$Y, t='n', main = "tsne")
text(tsne$Y, labels = train_data$label, col = colors[train_data$label])
# But it cannot generate the plot shown in the tutorial, I guess the tutorial is using another dataset for visualization
