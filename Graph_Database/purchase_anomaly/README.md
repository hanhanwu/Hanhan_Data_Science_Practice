# Purchase Anomaly

Here, I'm trying to use Graph DB to build a mini DB to show clients' purchasing behaviors, and see whether it's efficient to query the purchasing anomalies.

## Code & Display
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


[1]:https://github.com/hanhanwu/Hanhan_Data_Science_Practice/blob/master/Graph_Database/purchase_anomaly/purchasing_anomaly_cypher.cql
