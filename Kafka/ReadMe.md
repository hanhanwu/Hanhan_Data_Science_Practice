# Kafka Practice

## About Apach Kafka
* Your company is maintaining ,ultiple systems to queue the data, all of which has its bugs and limitations. You need a single centralized system that allows for publishing generic types of data, which will grow as your business grows. Kafka is <b>a publish/subscribe messaging system</b> designed to solve this problem.
### Components
* Message - the unit of data within Kafka, it's an array of bytes, imagine it as a record/row in a database.
* Key - Used to determine which partition the message to go, so that messages with the same key will be written into the same partition. It's also a byte array.
* Batch - A collections of messages with the same topic and partition. There's a tradeoff between latency and throughput, larger the batches, the more messages can be handled per unit of time, but the longer it takes an individual message to propagate. Batches are usually compressed for the efficiency of data transfer and storage.
