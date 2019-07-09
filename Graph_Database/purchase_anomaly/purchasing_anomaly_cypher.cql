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
    (Alice)-[:PLACED]->(d4)-[:CONTAINS {quantity:100}]->(prod4)

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
