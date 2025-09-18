For Spark 1.x, you can set with :

System.setProperty("hive.metastore.uris", "thrift://METASTORE:9083");
            
final SparkConf conf = new SparkConf();
SparkContext sc = new SparkContext(conf);
HiveContext hiveContext = new HiveContext(sc);
Or

final SparkConf conf = new SparkConf();
SparkContext sc = new SparkContext(conf);
HiveContext hiveContext = new HiveContext(sc);
hiveContext.setConf("hive.metastore.uris", "thrift://METASTORE:9083");




Try setting these before creating the HiveContext :

System.setProperty("hive.metastore.sasl.enabled", "true");
System.setProperty("hive.security.authorization.enabled", "false");
System.setProperty("hive.metastore.kerberos.principal", hivePrincipal);
System.setProperty("hive.metastore.execute.setugi", "true");
