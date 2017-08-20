# review Apriori here

library(arules)
library(arulesViz)
library(mlr)
library(RColorBrewer)

data("Groceries")
str(Groceries)
summarizeColumns(Groceries@itemInfo)
head(Groceries@itemInfo)    # from level 1 to level 2 to labels, the categories become more detailed
summary(Groceries)

rules <- apriori(Groceries, parameter = list(supp = 0.001, conf = 0.79))
inspect(rules[1:10])   # top 10 association rules


# plot relative item frequency
## you can place items around most frequently purchased products to increase their sales
arules::itemFrequencyPlot(Groceries,topN=20,
                          col=brewer.pal(8,'Set3'),   # create some nice color
                          main='Relative Item Frequency Plot',
                          type="relative",ylab="Item Frequency (Relative)")


# plot associate between products
## node size represent Support level
## color of nodes represent Life ratios
## you can put those strongly associated products together
plot(rules[1:10],
     method = "graph",
     control = list(type = "items"))


# plot visualizaed recommendation about what to buy together
## For example, here if you buy yogurt, it will recommend you to buy jam together (god, that's too sweet)
plot(rules[1:10], method = "paracoord", control = list(reorder = TRUE))
## amtrix version of the above, but it's really confusing...
plot(rules[1:10], method = "matrix", control = list(reorder = TRUE))


# Interactive visualization
## You can see, association rules, lift, support and confidence when mouse over the point
arulesViz::plotly_arules(rules)
