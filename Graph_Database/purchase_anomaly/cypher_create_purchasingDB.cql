// Clients
CREATE (Alice:Client {ip: '1.1.1.1', shipping_address: 'a place', billing_address: 'a place'})
CREATE (Bob:Client {ip: '1.1.1.2', shipping_address: 'b place', billing_address: 'b place'})
CREATE (Cindy:Client {ip: '1.1.1.3', shipping_address: 'c place', billing_address: 'c place'})
CREATE (Diana:Client {ip: '1.1.1.4', shipping_address: 'd place', billing_address: 'd place'})
CREATE (Emily:Client {ip: '1.1.1.5', shipping_address: 'e place', billing_address: 'e place'})
CREATE (Fiona:Client {ip: '1.1.1.6', shipping_address: 'f place', billing_address: 'f place'})

// Products
CREATE (prod1:Product {name: 'strawberry ice-cream', category: 'ice-cream', unit: 'box'})
CREATE (prod2:Product {name: 'mint ice-cream', category: 'ice-cream', unit: 'box'})
CREATE (prod3:Product {name: 'mango ice-cream', category: 'ice-cream', unit: 'box'})
CREATE (prod4:Product {name: 'cheesecake ice-cream', category: 'ice-cream', unit: 'box'})
CREATE (prod5:Product {name: 'orange', category: 'furit', unit: 'lb'})
CREATE (prod6:Product {name: 'dragon fruit', category: 'furit', unit: 'lb'})
CREATE (prod7:Product {name: 'kiwi', category: 'furit', unit: 'lb'})
CREATE (prod8:Product {name: 'cherry', category: 'furit', unit: 'lb'})
CREATE (prod9:Product {name: 'strawberry', category: 'furit', unit: 'lb'})
CREATE (prod10:Product {name: 'cherry', category: 'furit', unit: 'lb'})

// Orders
CREATE (d1:Order {id:'d1', deliverdate:'20190410', status:'delivered'})
CREATE (d2:Order {id:'d2', deliverdate:'20130708', status:'delivered'})
CREATE (d3:Order {id:'d3', deliverdate:'20021201', status:'delivered'})
CREATE (d4:Order {id:'d4', deliverdate:'20040612', status:'delivered'})
CREATE (d5:Order {id:'d5', deliverdate:'20110801', status:'delivered'})
CREATE (d6:Order {id:'d6', deliverdate:'20171212', status:'delivered'})

// Link Clients, Orders and ProductsCREATE
CREATE
    (Alice)-[:PLACED]->(d1)-[:CONTAINS {quantity:1}]->(prod1),
    (Alice)-[:PLACED]->(d1)-[:CONTAINS {quantity:2}]->(prod2)


//query results
MATCH (Alice)-[:PLACED]->(d1)-[:CONTAINS]->(prod1)
return prod1.category, prod1.unit;
