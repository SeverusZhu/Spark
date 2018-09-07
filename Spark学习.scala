meituan 2018-08-23

Spark基础文档

与Hadoop MapReduce计算框架相比，Spark所采用的Executor有两个优点：
一是利用多线程来执行具体的任务（Hadoop MapReduce采用的是进程模型），减少任务的启动开销；
二是Executor中有一个BlockManager存储模块，会将内存和磁盘共同作为存储设备，当需要多轮迭代计算时，
	可以将中间结果存储到这个存储模块里，下次需要时，就可以直接读该存储模块里的数据，而不需要读写到HDFS等文件系统里，因而有效减少了IO开销；
	或者在交互式查询场景下，预先将表缓存到该存储系统上，从而可以提高读写IO性能。


RDD 执行过程(RDD 为惰性调用， RDD 具有高效的容错性，中间结果持久化到内存避免不必要的读写磁盘开销，存放的数据可以是java对象，避免了不必要的对象序列化和反序列化开销)
1.  RDD读入外部数据源（或者内存中的集合）进行创建；
2.  RDD经过一系列的“转换”操作，每一次都会产生不同的RDD，供给下一个“转换”使用；
3.  最后一个RDD经“行动”操作进行处理，并输出到外部数据源（或者变成Scala集合或标量）。


// 创建SparkContext对象
val sc= new SparkContext(“spark://localhost:7077”,”Hello World”, “YOUR_SPARK_HOME”,”YOUR_APP_JAR”) 
// 从HDFS文件中读取数据创建一个RDD
val fileRDD = sc.textFile(“hdfs://192.168.0.103:9000/examplefile”) 
// 对fileRDD进行转换操作得到一个新的RDD，即filterRDD
val filterRDD = fileRDD.filter(_.contains(“Hello World”)) 
// 对filterRDD进行持久化，把它保存在内存或磁盘中（这里采用cache接口把数据集保存在内存中），方便后续重复使用，当数据被反复访问时（比如查询一些热点数据，或者运行迭代算法），
// 这是非常有用的，而且通过cache()可以缓存非常大的数据集，支持跨越几十甚至上百个节点
filterRDD.cache() 
// 计算一个RDD集合中包含的元素个数
filterRDD.count() 

程序的执行过程如下：

创建这个Spark程序的执行上下文，即创建SparkContext对象；

从外部数据源（即HDFS文件）中读取数据创建fileRDD对象；

构建起fileRDD和filterRDD之间的依赖关系，形成DAG图，这时候并没有发生真正的计算，只是记录转换的轨迹；

执行到第5行代码时，count()是一个行动类型的操作，触发真正的计算，开始实际执行从fileRDD到filterRDD的转换操作，并把结果持久化到内存中，最后计算出filterRDD中包含的元素个数。


RDD 运行过程
 (1) 创建RDD对象；
（2）SparkContext负责计算RDD之间的依赖关系，构建DAG；
（3）DAGScheduler负责把DAG图分解成多个阶段，每个阶段中包含了多个任务，
	每个任务会被任务调度器分发给各个工作节点（Worker Node）上的Executor去执行。
参见 RDD在Spark中的运行过程.jpg


/***********************************************************************************************************************************/
/***********************************************************************************************************************************/

RDD 创建 
1. 读取外部数据集
	从文件系统中加载数据创建RDD
scala> val lines = sc.textFile("file:///usr/local/spark/mycode/rdd/word.txt")
lines: org.apache.spark.rdd.RDD[String] = file:///usr/local/spark/mycode/rdd/word.txt MapPartitionsRDD[12] at textFile at <console>:27	

从HDFS文件系统中加载数据：（下面三条等价）
scala> val lines = sc.textFile("hdfs://localhost:9000/user/hadoop/word.txt")
scala> val lines = sc.textFile("/user/hadoop/word.txt")
scala> val lines = sc.textFile("word.txt")


如果使用了本地文件系统的路径，
	必须要保证在所有的worker节点上，也都能够采用相同的路径访问到该文件，
	比如，可以把该文件拷贝到每个worker节点上，或者也可以使用网络挂载共享文件系统。

textFile()方法的输入参数，可以是文件名，目录，压缩文件等。
	比如，textFile(“/my/directory”), textFile(“/my/directory/*.txt”), and textFile(“/my/directory/*.gz”).

textFile()方法也可以接受第2个输入参数（可选），用来指定分区的数目。
	默认情况下，Spark会为HDFS的每个block创建一个分区（HDFS中每个block默认是128MB）。
	你也可以提供一个比block数量更大的值作为分区数目，但是，你不能提供一个小于block数量的值作为分区数目。


2. 调用SparkContext的parallelize方法，在Drive中一个已经存在的集合（数组）上创建

使用数组来创建
scala>val array = Array(1,2,3,4,5)
array: Array[Int] = Array(1, 2, 3, 4, 5)
scala>val rdd = sc.parallelize(array)
rdd: org.apache.spark.rdd.RDD[Int] = ParallelCollectionRDD[13] at parallelize at <console>:29


从列表中创建
scala>val list = List(1,2,3,4,5)
list: List[Int] = List(1, 2, 3, 4, 5)
scala>val rdd = sc.parallelize(list)
rdd: org.apache.spark.rdd.RDD[Int] = ParallelCollectionRDD[14] at parallelize at <console>:29





