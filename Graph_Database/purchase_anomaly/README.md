# Purchase Anomaly

Here, I'm trying to use Graph DB to build a mini DB to show clients' purchasing behaviors, and see whether it's efficient to query the purchasing anomalies.

## Cypher Code & Display
* To find all the Cypher Query [here][1]
### Purchasing DB
* First of all, you need to create this grapg DB, there are clients, orders and products. Products have prices and their quantities are in the realtionship "CONTAINS".
<img src="https://github.com/hanhanwu/Hanhan_Data_Science_Practice/blob/master/Graph_Database/purchase_anomaly/images/purchase_DB.png" width="80%">

### Alice's Orders
* Here're all the orders Alice has placed.
<img src="https://github.com/hanhanwu/Hanhan_Data_Science_Practice/blob/master/Graph_Database/purchase_anomaly/images/alice_orders.png" width="80%">

### Purchase History
* Historical Orders for current orders. It doesn't have direct relationship with purchase anomaly detection below, but it's a new way to use something similar to "for" loop and `UNWIND` to expand the list.
  * Note that before `UNWIND`, there is no comma
<img src="https://github.com/hanhanwu/Hanhan_Data_Science_Practice/blob/master/Graph_Database/purchase_anomaly/images/purchase_history.png" width="80%">

### Purchase Anomaly
* It calculates the standard deviation of other orders' total cost, so when the standard deviate suddenly changed a lot, you will know the above order is an anomaly. In this case, order 4 is the anomaly
<img src="https://github.com/hanhanwu/Hanhan_Data_Science_Practice/blob/master/Graph_Database/purchase_anomaly/images/purchase_anomaly.png" width="80%">


## Python Neo4j Version
### [Python version code][2]
* To install the library, just type `pip install neo4j` in your terminal.
* Better not to install "Py2neo" afer installing "neo4j", Py2neo might break multiple preinstalled packages. If you really want to try Py2neo, better to use virtual environment
* There is also a tutorial for [neo4jrestapi][3], I personally prefer "neo4j" than this library, since you can write nodes, labels and properties in 1 query, while this library needs you to write them seperately, less convenient.
* When you are running this code, you can also query everything through http://localhost:7474/browser/
  * Here's all the orders and products each client had:
    <img src="https://github.com/hanhanwu/Hanhan_Data_Science_Practice/blob/master/Graph_Database/purchase_anomaly/images/python_neo4j_browser_visual.PNG" width="70%">
  
  * Same for seeing anomaly order:
  <img src="https://github.com/hanhanwu/Hanhan_Data_Science_Practice/blob/master/Graph_Database/purchase_anomaly/images/python_neo4j_anomalies_browser_visual.PNG" width="70%">
    
### Resource Links
  * [What you can output from BoltStatementResult][4]

[1]:https://github.com/hanhanwu/Hanhan_Data_Science_Practice/blob/master/Graph_Database/purchase_anomaly/purchasing_anomaly_cypher.cql
[2]:https://github.com/hanhanwu/Hanhan_Data_Science_Practice/blob/master/Graph_Database/purchase_anomaly/purchase%20anomaly.ipynb
[3]:https://marcobonzanini.com/2015/04/06/getting-started-with-neo4j-and-python/
[4]:https://neo4j.com/docs/api/python-driver/current/results.html
