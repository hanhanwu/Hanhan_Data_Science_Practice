match (n) detach delete n;

// Clients
CREATE (Alice:Client {name:'Alice', ip: '1.1.1.1', shipping_address: 'a place', billing_address: 'a place'})
CREATE (Bob:Client {name:'Bob', ip: '1.1.1.2', shipping_address: 'b place', billing_address: 'b place'})
CREATE (Cindy:Client {name:'Cindy', ip: '1.1.1.3', shipping_address: 'c place', billing_address: 'c place'})
CREATE (Diana:Client {name:'Diana', ip: '1.1.1.4', shipping_address: 'd place', billing_address: 'd place'})
CREATE (Emily:Client {name:'Emily', ip: '1.1.1.5', shipping_address: 'e place', billing_address: 'e place'})
CREATE (Fiona:Client {name:'Fiona', ip: '1.1.1.6', shipping_address: 'f place', billing_address: 'f place'})

// Products
CREATE (prod1:Product {name: 'strawberry ice-cream', category: 'ice-cream', price: 6.9, unit: 'box'})
CREATE (prod2:Product {name: 'mint ice-cream', category: 'ice-cream', price: 6.9, unit: 'box'})
CREATE (prod3:Product {name: 'mango ice-cream', category: 'ice-cream', price: 6.9, unit: 'box'})
CREATE (prod4:Product {name: 'cheesecake ice-cream', category: 'ice-cream', price: 7.9, unit: 'box'})
CREATE (prod5:Product {name: 'orange', category: 'furit', unit: 'lb', price: 2.6, unit: 'box'})
CREATE (prod6:Product {name: 'dragon fruit', category: 'furit', unit: 'lb', price: 4.8, unit: 'box'})
CREATE (prod7:Product {name: 'kiwi', category: 'furit', unit: 'lb', price: 5.3, unit: 'box'})
CREATE (prod8:Product {name: 'cherry', category: 'furit', unit: 'lb', price: 4.8, unit: 'box'})
CREATE (prod9:Product {name: 'strawberry', category: 'furit', unit: 'lb', price: 3.9, unit: 'box'})

// Orders
CREATE (d1:Order {id:'d1', name:'d1', deliverdate:'20190410', status:'delivered'})
CREATE (d2:Order {id:'d2', name:'d2', deliverdate:'20130708', status:'delivered'})
CREATE (d3:Order {id:'d3', name:'d3', deliverdate:'20021201', status:'delivered'})
CREATE (d4:Order {id:'d4', name:'d4', deliverdate:'20040612', status:'delivered'})
CREATE (d5:Order {id:'d5', name:'d5', deliverdate:'20110801', status:'delivered'})
CREATE (d6:Order {id:'d6', name:'d6',deliverdate:'20171212', status:'delivered'})

// Link Clients, Orders and ProductsCREATE
CREATE
    (Alice)-[:PLACED]->(d1)-[:CONTAINS {quantity:1}]->(prod1),
    (d1)-[:CONTAINS {quantity:2}]->(prod2),
    (Bob)-[:PLACED]->(d2)-[:CONTAINS {quantity:2}]->(prod1),
    (d2)-[:CONTAINS {quantity:6}]->(prod7),
    (Cindy)-[:PLACED]->(d3)-[:CONTAINS {quantity:1}]->(prod9),
    (Alice)-[:PLACED]->(d4)-[:CONTAINS {quantity:100}]->(prod4),
    (Alice)-[:PLACED]->(d5)-[:CONTAINS {quantity:10}]->(prod8),
    (Alice)-[:PLACED]->(d6)-[:CONTAINS {quantity:1}]->(prod7);
  

//Display all clients, orders, and products graph
MATCH (c:Client)-[:PLACED]-(o)-[:CONTAINS]->(p)
return c, o, p;

//Display a certain client's order and products graph
MATCH (c:Client)-[:PLACED]-(o)-[:CONTAINS]->(p)
where c.name='Alice'
return c, o, p;

// Show certain clients' products price
MATCH (c:Client)-[:PLACED]-(o)-[cts:CONTAINS]->(p)
WHERE c.name='Alice'
return distinct c.name, o.id, cts.quantity, p.price, cts.quantity * p.price as prod_price
UNION
MATCH (c)-[:PLACED]-(o)-[cts:CONTAINS]->(p)
WHERE c.name='Bob'
return distinct c.name, o.id, cts.quantity, p.price, cts.quantity * p.price as prod_price;

// Find abnormal purchase behavior
MATCH (c:Client)-[:PLACED]-(o)-[cts:CONTAINS]->(p)
where cts.quantity > 90
return distinct c.name, o.id, cts.quantity, p.price, cts.quantity * p.price as prod_price;

// Aggregated results for each order
MATCH (c:Client)-[:PLACED]-(o)-[cts:CONTAINS]->(p)
WHERE c.name = 'Alice'
RETURN c.name, o.id, COLLECT(p.name) as products, 
COLLECT(cts.quantity) as quantities, COLLECT(p.price) as prices, 
COLLECT(cts.quantity * p.price) as total_cost, sum(cts.quantity*p.price) as order_cost;

// Count the number of orders, products per Client
MATCH (c:Client)-[:PLACED]-(o)-[cts:CONTAINS]->(p)
RETURN c.name, collect(distinct o.id) as orders, collect(distinct p.name) as products,
size(COLLECT(distinct o.id) ) as num_of_orders, 
size(COLLECT(distinct p.name)) as num_of_prods;

// order Alice's order's by time
MATCH (c:Client)-[:PLACED]-(o)-[:CONTAINS]->(p)
where c.name='Alice'
return c.name, o.id, o.deliverdate, collect(p.name) as products
order by o.deliverdate;

// orders and product list for Alice
MATCH (c:Client)-[:PLACED]-(o)-[:CONTAINS]->(p)
where c.name='Alice'
return c.name, o.id, o.deliverdate, collect(p.name) as products
order by o.deliverdate;

// find historical orders for each current order
MATCH (c:Client {name:'Alice'})-[:PLACED]->(o)
WITH c, o ORDER BY o.deliverdate
WITH c.name AS name, COLLECT(o) AS os
UNWIND [i IN RANGE(0, SIZE(os)-1) |
  {name: name, id: os[i].id, history: [x IN os[0..i] | x.id]}] AS result
RETURN result.name, result.id as order_id, result.history;


// find historical orders for each current order, also sum up current orders
MATCH (c:Client {name:'Alice'})-[:PLACED]->(o)-[cts:CONTAINS]->(p)
WITH c, o, COLLECT(cts.quantity) as prod_quantity_lst, COLLECT(p.price) as prod_price_lst, 
  SUM(cts.quantity * p.price) as order_price
  ORDER BY o.deliverdate
WITH c.name AS name, COLLECT(o) AS os,
  COLLECT(prod_quantity_lst) as qss, COLLECT(prod_price_lst) AS ppss,
  COLLECT(order_price) as ops
UNWIND [i IN RANGE(0, SIZE(os)-1) |
  {name: name, id: os[i].id, prod_quanty_lst: qss[i], prod_price_lst: ppss[i], 
  order_price: ops[i],
  history: [x IN os[0..i] | x.id]}] AS result
RETURN result.name as name, result.id as order_id, 
result.prod_quanty_lst as prod_quanty_lst, result.prod_price_lst as prod_price_lst, 
round(result.order_price) as order_price, result.history as order_history;

// find historical orders, and the total price for all the historical orders
MATCH (c:Client {name:'Alice'})-[:PLACED]->(o)-[cts:CONTAINS]->(p)
WITH c, o, SUM(cts.quantity * p.price) as order_price
  ORDER BY o.deliverdate
WITH c.name AS name, COLLECT(o) AS os, COLLECT(order_price) as ops
UNWIND [i IN RANGE(0, SIZE(os)-1) |
  {name: name, id: os[i].id, order_price: ops[i],
  historical_orders: [x IN os[0..i] | x.id],
  historical_order_costs: [x IN ops[0..i+1] | round(x)]}] AS result
return result.name, result.id as order_id, result.historical_orders, 
result.historical_order_costs;

// collect other orders for current order
MATCH (c:Client {name:'Alice'})-[:PLACED]->(o)-[cts:CONTAINS]->(p)
WITH c, o, SUM(cts.quantity * p.price) as order_price
  ORDER BY o.deliverdate
WITH c.name AS name, COLLECT(o) AS os, COLLECT(order_price) as ops
UNWIND [i IN RANGE(0, SIZE(os)-1) |
  {name: name, id: os[i].id, order_price: ops[i],
  other_orders: [x IN os[0..i] + os[i+1..SIZE(os)] | x.id]
  }] AS result
return result.name, result.id as order_id, result.other_orders;


// collect other orders for current order, and compare current order price with other orders' total price
MATCH (c:Client {name:'Alice'})-[:PLACED]->(o)-[cts:CONTAINS]->(p)
WITH c, o, SUM(cts.quantity * p.price) as order_price
  ORDER BY o.deliverdate
WITH c.name AS name, COLLECT(o) AS os, COLLECT(order_price) as ops
UNWIND [i IN RANGE(0, SIZE(os)-1) |
  {name: name, id: os[i].id, current_order_cost: round(ops[i]),
  other_orders: [x IN os[0..i] + os[i+1..SIZE(os)] | x.id],
  other_orders_costs: [x IN ops[0..i] + ops[i+1..SIZE(os)] | round(x)]
  }] AS result
WITH result.name as name, result.id as order_id, result.current_order_cost as current_order_cost,
result.other_orders as other_orders, result.other_orders_costs as other_orders_costs
UNWIND(other_orders_costs) as unwind_other_orders_costs
return name, order_id, current_order_cost, other_orders, other_orders_costs,
round(stDev(unwind_other_orders_costs)) as other_costs_std;
