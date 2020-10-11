# Kafka Practice

## About Apach Kafka
* Your company is maintaining ,ultiple systems to queue the data, all of which has its bugs and limitations. You need a single centralized system that allows for publishing generic types of data, which will grow as your business grows. Kafka is <b>a publish/subscribe messaging system</b> designed to solve this problem.
* [Technical Document][2]
* [Definitive Guide][4]
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
* Schema Registry
  * Avro requires to present the entire schema when reading the record, but there is too much overhead when storig the entire shcema in each record. 
  * Therefore we can locate the entire schema in a schema registry, such as [Confluent Schema Registry][5]
    * [How to use COnfluent Shcema Registry][6]

### Java Coding Notes
#### Serializer
* Recommended to use Apach existing serializer, such as Apach Avro
  * NOT kafka built-in serializer, since they are data type specific
  * Self implemented generic serializer is not flexible to maintain, when there is schema change, there will be error
* Serializers like Avro allows the change of data schema without exception or breaking errors, no need expensive update either
  * However the schema used for writing and reading should be compatible, [here are some compatibility rules][3]
* 2 Options when using schema registry and serializer
  * Option 1 - Specialized Avro Object
    * Need to define the schema and [create a specialized Avro object][7]
    * Avro serrializer cannot process regular Java class but has to be Avro object
  * Option 2 - Generic Avro Object
    * Same way of using schema registry as option 1, but need to provide the schema as a string in the code
    * The generic type here can be [Avro GenricRecord][8] - A generic instance of a record schema.
      * [All the avro generic][9]

  
### [Clients written in other languages][1]
* The clients are like APIs used to interact with Kafka

[1]:https://cwiki.apache.org/confluence/display/KAFKA/Clients
[2]:http://kafka.apache.org/documentation.html#gettingStarted
[3]:https://avro.apache.org/docs/1.7.7/spec.html#Schema+Resolution
[4]:https://book.huihoo.com/pdf/confluent-kafka-definitive-guide-complete.pdf
[5]:https://github.com/confluentinc/schema-registry
[6]:https://docs.confluent.io/current/schema-registry/index.html
[7]:http://avro.apache.org/docs/current/gettingstartedjava.html
[8]:https://avro.apache.org/docs/1.7.6/api/java/org/apache/avro/generic/GenericRecord.html
[9]:https://avro.apache.org/docs/1.7.6/api/java/org/apache/avro/generic/package-summary.html
