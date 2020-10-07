# Kafka Practice

## About Apach Kafka
* Your company is maintaining ,ultiple systems to queue the data, all of which has its bugs and limitations. You need a single centralized system that allows for publishing generic types of data, which will grow as your business grows. Kafka is <b>a publish/subscribe messaging system</b> designed to solve this problem.
* [Technical Document][2]
### Components
* Messages & Batches
* Schemas - consistent data format is important in Kafka
* Topics & Partitions
* Producers & Consumers
* Brokers & Clusters
  * a broker is a single Kafka server
  * multiple Clusters is better
### Tools
* Zookeeper: It stores the metadata for Kafka cluster and consumer 
* Kafka Broker (server)

### Java Coding Notes
#### Serializer
* Recommended to use Apach existing serializer, such as Apach Avro
  * NOT kafka built-in serializer, since they are data type specific
  * Self implemented generic serializer is not flexible to maintain, when there is schema change, there will be error
* Serializers like Avro allows the change of data schema without exception or breaking errors, no need expensive update either
  * However the schema used for writing and reading should be compatible, [here are some compatibility rules][3]
### [Clients written in other languages][1]
* The clients are like APIs used to interact with Kafka

[1]:https://cwiki.apache.org/confluence/display/KAFKA/Clients
[2]:http://kafka.apache.org/documentation.html#gettingStarted
[3]:https://avro.apache.org/docs/1.7.7/spec.html#Schema+Resolution
