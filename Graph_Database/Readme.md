
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
* Index typically takes `O(log(n))` time to traverse a physical relationship, but index free graph just needs `O(1)` time, since you just need to follow the incoming or outgoing relationships of a node.
* So in index free graph DB, the time of the query is NOT decided by the total data size of the databse as relational DB does, but decided by the data being queried.
* In most graph DB, most queries follow a pattern whereby an index is used simply to find a starting node, the remainder of the traversal then uses a combination of pointer chasing and pattern matching to search the data store.
* The physical structure of graph database is different from the visualized graph strucuture. 
* <b>However, adding index in `MATCH` clause in Cypher, can improve lookup performance.</b>
  * With the index here, you can pick out specific nodes directly, as the lookup staring point.
  * Cypher allows to create indexs per label and property combinations.
### Some Concepts in Graph Theory
* DFS & BFS
  * Breath-first Search is good to search for paths
  * When the search won't stop until it reaches to the end, it's called uninformed search. With informed search, the stop can happen ealrlier, which can be more efficient
* Dijkstra's Algorithm
  * It's used to find the shortest path between 2 nodes.
  * It's using Breath First search + Best First search. The problem with this method is, it can follow nodes that will never controbute to the final shortest path, which wastes more time.
  * Time Efficiency: `O(R + N*log(N))`, R is the number of relationship, N is the number of nodes
* A* Algorithms
  * It's improved on Dijkstra's Algorithm:
    * `g(n)` measures the cost from starting point to node n
    * `h(n)` measures the cost from node n to the destination
    * It chooses the nodes with lowest `g(n) + h(n)`
* Triadic Closure
  * A triadic closure is a common property of social graphs, where we observe that if two nodes are connected via a path involving a third node, there is an increased likelihood that the two nodes will become directly connected at some point in the future.
  * There are strong & weak relationship in it, our preference for symmetry and rational layering formed structural balance.
### Notes from Real World Graph DB Applications
* Precompute new kinds of relationships [Improve Query Efficiency]
  * It helps to enrich the shortcuts for performance-critical access patterns
  * It’s quite common to optimize graph access by adding a direct relationship between two nodes that would otherwise be connected only by way of intermediaries.

### Cypher
* Quick Cypher Cheatsheet: https://gist.github.com/DaniSancas/1d5265fc159a95ff457b940fc5046887
* Introduction: https://neo4j.com/developer/cypher-query-language/
* Cypher Clauses: https://neo4j.com/docs/cypher-manual/current/clauses/
* Data Sample & Data Export, Import Queries: https://neo4j.com/developer/guide-importing-data-and-etl/
* SQL vs Cypher: https://neo4j.com/developer/guide-sql-to-cypher/
  * `IN ['a', 'b']`, IN clause
  * `like` replaced by `STARTS WITH`, `CONTAINS`, `ENDS WITH`
  * Connnections between nodes can be used as `join`
  * `MATCH` together with `OPTIONAL MATCH` are used as `OUTER JOIN`, you just change the position of entities, it will serve as left outer join or right outer join.
  * Group by is automated as long as you use Cypher [Aggregated Functions][1]
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
  * Using `with` will chain together several `MATCH`, since sometimes it's not possible to do everything in 1 `MATCH`
    * While `match` is to do the search, `with` put those "columns" together in the search results.
    * Using `collect()` in return clause, the results will be displayed as a list in 1 row, delimited by comma
    * When there is `where`, use `with` after `where`
    * It seems that in `WITH` clause, you cannot call the properties of the entities or repationships.
  * `MERGE` is like a mixture of MATCH and CREATE. If the pattern described in the MERGE statement already exists in the graph, the statement’s identifiers will be bound to this existing data, much as if we’d specified MATCH. If the pattern does not currently exist in the graph, MERGE will create it, much as if we’d used CREATE.
    * Using `MERGE` to match existing pattern, if it can’t match all parts of a pattern, MERGE will create a new instance of the entire pattern, which could lead to data duplication. So it's better to break apart the larger pattern into smaller chunks when using `merge` to match.
  * `= null`, lower case, `= true` also lower case.
  * Properties used in `order by` have to appear in `return` clause too
### Sample Cypher & Graph
<b>I'm using neo4j console, it's simple to use and has visualized graph generated.</b>
#### The Name of Nodes
* When you are trying to return nodes that all belong to a certain Entity, it's better to have "name" property in each node so that the returned nodes will show you the name, otherwise it can be a random property value.
* For example:
The "name" property here is important. When using `match (c:Client) return c`, it will return names by default, otherwise will be a random property value, which can be confusing. Even "id" property won't work.
```
// Clients
CREATE (Alice:Client {name:'Alice', ip: '1.1.1.1', shipping_address: 'a place', billing_address: 'a place'})
CREATE (Bob:Client {name:'Bob', ip: '1.1.1.2', shipping_address: 'b place', billing_address: 'b place'})
CREATE (Cindy:Client {name:'Cindy', ip: '1.1.1.3', shipping_address: 'c place', billing_address: 'c place'})
CREATE (Diana:Client {name:'Diana', ip: '1.1.1.4', shipping_address: 'd place', billing_address: 'd place'})
CREATE (Emily:Client {name:'Emily', ip: '1.1.1.5', shipping_address: 'e place', billing_address: 'e place'})
CREATE (Fiona:Client {name:'Fiona', ip: '1.1.1.6', shipping_address: 'f place', billing_address: 'f place'})

match (c:Client) return c
```
#### Create table
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

#### Match with Depth
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

#### Shortest Path
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


[1]:https://neo4j.com/docs/cypher-manual/current/functions/aggregating/index.html
