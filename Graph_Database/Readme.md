
## Graph Libraries
* [Neo4j][1]
* [Boost Graph Library][2] - this is more about an interface of graph implementation
* [iGraph][5]
  * It supports both R and python
  * R package: https://igraph.org/r/#startr
  * Python package: https://igraph.org/python/
* [NetworkX][6]
* [Graph Tool][3]
  * It's a python library written in C++, so should be fast
  * [Its documentation][4]


[1]:https://neo4j.com/developer/get-started/
[2]:https://www.boost.org/doc/libs/1_70_0/libs/graph/doc/index.html
[3]:https://graph-tool.skewed.de/
[4]:https://graph-tool.skewed.de/static/doc/index.html
[5]:https://igraph.org/
[6]:https://github.com/networkx/networkx

## Tutorial
### Neo4j Graph DB
* URL: https://github.com/iansrobinson/graph-databases-use-cases
* How to intsall and open Neo4j console locally: https://github.com/hanhanwu/Hanhan_Data_Science_Practice/blob/master/README.md#graph-database-with-neo4j
### 3 Types of Graph DB
* Native graph DBs do not depend heavily on index, since the graph itself provides natural adjacency index. But not all graph DB is native graph DB. 
  * Graph queries use its natural adjacency index to traverse through the graph by chasing pointers. These operations can be carried out with extreme efficiency, tra‐ versing millions of nodes per second, in contrast to joining data through a global index, which is many orders of magnitude slower.
* Type 1 - Property Graph
  * What we often see, nodes and relationships, both nodes and relationships can have properties, nodes can have labels, and relationships can have directions.
* Type 2 - Hypergraph
  * Used in many-to-many relatioships, both source and destionation of a relationship can have as many nodes it needs.
  * Property Graph and Hypergraph are isomophic, you can transform a hypergraph into a property graph
* Type 3 - Triples
  * It came from Semantic Web movement, where researchers are interes‐ ted in large-scale knowledge inference by adding semantic markup to the links that connect web resources.
  * A triple is a subject-predicate-object data structure. 
    * Using triples, we can capture facts, such as “Ginger dances with Fred” and “Fred likes ice cream.”
### Index Free Graph DB vs Relational DB
#### Cypher
* Introduction: https://neo4j.com/developer/cypher-query-language/
* Notes
  * Graph Characteristics (node, property, label, relationship)
    * It contains nodes and relationships.
    * Nodes contain properties (key-value pairs).
    * Nodes can be labeled with one or more labels.
    * Relationships are named and directed, and always have a start and end node.
    * Relationships can also contain properties.
  * When using `MATCH` to do the search query, you need to specify the starting point. If the property of the label has an index, it will be faster to do the search, especially in a large database.
  * Pattern Nodes - to restrict the pattern when there are multiple records
    * It's a node, doesn't need `where`, although serves like `where`
    * eg. `(newcastle)<-[:STREET|CITY*1..2]-theater`, it means from theater node there can be no more than 2 street-city relations go out to newcastle node, because in the database there can be mutliple theaters have the same name and located in different cities, this constraints is trying to locate the threater in Newcastle
      * The concept here has something similar to cardinality, and it can be difficult to understand at the very beginning...even though the book said with graph DB you don't need to worry about cardinality as relational DB...
  * Anonymous Nodes - the node doesn't specify label and property
    * You use `()` as a node in match when you don't care any detail about the node.
    * If you want to return the values of anonymous nodes, you can have node name in it, such as `(product)` to return the "products" of your search
  * Using `with` will chain returned results together
    * I think `match` and `with` together plays a role as `select` in relational database, while `match` is to do the search, `with` put those "columns" together in the search results.
    * Using `collect()` in return clause, the results will be displayed as a list in 1 row, delimited by comma
    * When there is `where`, use `with` after `where`
#### Sample Cypher & Graph
##### I'm using neo4j console, it's simple to use and has visualized graph generated.
* Create table
  * In this code, I chose a very small part of the sample query. In console, each time you can only run 1 query, which means there should only be 1 `;`.
  * In order to show the visualized graph, only `create` is not enough, you need `match` too.
  * The query here is to find directors of the film that Keanu was acted in.
```sql
CREATE (TheMatrix:Movie {title:'The Matrix', released:1999, tagline:'Welcome to the Real World'})
CREATE (Keanu:Person {name:'Keanu Reeves', born:1964})
CREATE (Carrie:Person {name:'Carrie-Anne Moss', born:1967})
CREATE (Laurence:Person {name:'Laurence Fishburne', born:1961})
CREATE (Hugo:Person {name:'Hugo Weaving', born:1960})
CREATE (LillyW:Person {name:'Lilly Wachowski', born:1967})
CREATE (LanaW:Person {name:'Lana Wachowski', born:1965})
CREATE (JoelS:Person {name:'Joel Silver', born:1952})
CREATE
  (Keanu)-[:ACTED_IN {roles:['Neo']}]->(TheMatrix),
  (Carrie)-[:ACTED_IN {roles:['Trinity']}]->(TheMatrix),
  (Laurence)-[:ACTED_IN {roles:['Morpheus']}]->(TheMatrix),
  (Hugo)-[:ACTED_IN {roles:['Agent Smith']}]->(TheMatrix),
  (LillyW)-[:DIRECTED]->(TheMatrix),
  (LanaW)-[:DIRECTED]->(TheMatrix),
  (JoelS)-[:PRODUCED]->(TheMatrix)

WITH Keanu as a
MATCH (a)-[:ACTED_IN]->(m)<-[:DIRECTED]-(d) RETURN a,m,d LIMIT 10;
```
<p align="left">
<img width="270" height="270" src="https://github.com/hanhanwu/Hanhan_Data_Science_Practice/blob/master/Graph_Database/sample_graph1.png">
</p>

* Match with Depth
  * Left is the result with 2 or 3 depth; Right is the result with 4 depth
  * I added Carrie into another movie that Keanu acted in to form a connection between the 2 movies
```sql
CREATE (TheMatrix:Movie {title:'The Matrix', released:1999, tagline:'Welcome to the Real World'})
CREATE (Keanu:Person {name:'Keanu Reeves', born:1964})
CREATE (Carrie:Person {name:'Carrie-Anne Moss', born:1967})
CREATE (Laurence:Person {name:'Laurence Fishburne', born:1961})
CREATE (Hugo:Person {name:'Hugo Weaving', born:1960})
CREATE (LillyW:Person {name:'Lilly Wachowski', born:1967})
CREATE (LanaW:Person {name:'Lana Wachowski', born:1965})
CREATE (JoelS:Person {name:'Joel Silver', born:1952})
CREATE
  (Keanu)-[:ACTED_IN {roles:['Neo']}]->(TheMatrix),
  (Carrie)-[:ACTED_IN {roles:['Trinity']}]->(TheMatrix),
  (Laurence)-[:ACTED_IN {roles:['Morpheus']}]->(TheMatrix),
  (Hugo)-[:ACTED_IN {roles:['Agent Smith']}]->(TheMatrix),
  (LillyW)-[:DIRECTED]->(TheMatrix),
  (LanaW)-[:DIRECTED]->(TheMatrix),
  (JoelS)-[:PRODUCED]->(TheMatrix)

CREATE (TheReplacements:Movie {title:'The Replacements', released:2000, tagline:'Pain heals, Chicks dig scars... Glory lasts forever'})
CREATE (Brooke:Person {name:'Brooke Langton', born:1970})
CREATE (Gene:Person {name:'Gene Hackman', born:1930})
CREATE (Orlando:Person {name:'Orlando Jones', born:1968})
CREATE (Howard:Person {name:'Howard Deutch', born:1950})
CREATE
  (Keanu)-[:ACTED_IN {roles:['Shane Falco']}]->(TheReplacements),
  (Carrie)-[:ACTED_IN {roles:['Test Person']}]->(TheReplacements),
  (Brooke)-[:ACTED_IN {roles:['Annabelle Farrell']}]->(TheReplacements),
  (Gene)-[:ACTED_IN {roles:['Jimmy McGinty']}]->(TheReplacements),
  (Orlando)-[:ACTED_IN {roles:['Clifford Franklin']}]->(TheReplacements),
  (Howard)-[:DIRECTED]->(TheReplacements)

WITH Keanu as a
MATCH (a)-[*1..4]-(hollywood)
RETURN DISTINCT hollywood;
```
<p align="left">
<img width="370" height="270" src="https://github.com/hanhanwu/Hanhan_Data_Science_Practice/blob/master/Graph_Database/3_hops.png">
 <img width="370" height="270" src="https://github.com/hanhanwu/Hanhan_Data_Science_Practice/blob/master/Graph_Database/4_hops.png">
</p>

* Shortest Path
  * Find the shortest path that Hugo can reach to Gene.
```sql
CREATE (TheMatrix:Movie {title:'The Matrix', released:1999, tagline:'Welcome to the Real World'})
CREATE (Keanu:Person {name:'Keanu Reeves', born:1964})
CREATE (Carrie:Person {name:'Carrie-Anne Moss', born:1967})
CREATE (Laurence:Person {name:'Laurence Fishburne', born:1961})
CREATE (Hugo:Person {name:'Hugo Weaving', born:1960})
CREATE (LillyW:Person {name:'Lilly Wachowski', born:1967})
CREATE (LanaW:Person {name:'Lana Wachowski', born:1965})
CREATE (JoelS:Person {name:'Joel Silver', born:1952})
CREATE
  (Keanu)-[:ACTED_IN {roles:['Neo']}]->(TheMatrix),
  (Carrie)-[:ACTED_IN {roles:['Trinity']}]->(TheMatrix),
  (Laurence)-[:ACTED_IN {roles:['Morpheus']}]->(TheMatrix),
  (Hugo)-[:ACTED_IN {roles:['Agent Smith']}]->(TheMatrix),
  (LillyW)-[:DIRECTED]->(TheMatrix),
  (LanaW)-[:DIRECTED]->(TheMatrix),
  (JoelS)-[:PRODUCED]->(TheMatrix)

CREATE (TheReplacements:Movie {title:'The Replacements', released:2000, tagline:'Pain heals, Chicks dig scars... Glory lasts forever'})
CREATE (Brooke:Person {name:'Brooke Langton', born:1970})
CREATE (Gene:Person {name:'Gene Hackman', born:1930})
CREATE (Orlando:Person {name:'Orlando Jones', born:1968})
CREATE (Howard:Person {name:'Howard Deutch', born:1950})
CREATE
  (Keanu)-[:ACTED_IN {roles:['Shane Falco']}]->(TheReplacements),
  (Carrie)-[:ACTED_IN {roles:['Test Person']}]->(TheReplacements),
  (Brooke)-[:ACTED_IN {roles:['Annabelle Farrell']}]->(TheReplacements),
  (Gene)-[:ACTED_IN {roles:['Jimmy McGinty']}]->(TheReplacements),
  (Orlando)-[:ACTED_IN {roles:['Clifford Franklin']}]->(TheReplacements),
  (Howard)-[:DIRECTED]->(TheReplacements)

WITH Hugo as a
MATCH p=shortestPath(
  (a)-[*]-(Gene:Person {name:"Gene Hackman"})
)
RETURN p;
```
<p align="left">
<img width="370" height="270" src="https://github.com/hanhanwu/Hanhan_Data_Science_Practice/blob/master/Graph_Database/shortestpath.png">
</p>
