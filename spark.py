

# 以下部分内容参考  Spark 快速大数据分析



# initialize spark
from pyspark import SparkContext, SparkConf

conf = SparkConf().setMaster("local").setAppName("My App")   # Cluster URL: local     App name: My App
sc = SparkContext( conf = conf)

# use textFile() create a string RDD
lines = sc.textFile("README.md")    # 1st kind

# put RDD persist into the ram
pythonLines = lines.filter(lambda line: "python" in line)
pythonLines.persist
pythonLines.count()

lines = sc.parallelize(["pandas", "i like pandas"])    # 2nd kind

# filter
# pick error messages from log.txt
inputRDD = sc.textFile("log.txt")
errorsRDD = inputRDD.filter(lambda x: "error" in x)

warningsRDD = inputRDD.filter(lambda x: "warning" in x)

# union()
badLinesRDD = errorsRDD.union(warningsRDD)

# 在python中使用行动操作对错误进行计数
print "Input had" + badLinesRDD.count() + "concerning lines"
print "Here are 10 examples:"
for line in badLinesRDD.take(10):
	print line

# collect 不能用在大规模数据集上
# collect要求所有数据都必须能一同放到单台机器的内存中

# 使用python存在三种方式将函数传递给Spark
#       传递比较短函数时，使用lambda
#       传递顶层函数
#       定义的局部函数
word = rdd.filter(lambda s:"error" in s)

def containsError(s):
	return "error" in s

word = rdd.filter(containsError)

# 计算RDD中各值的平方
nums = sc.parallelize([1,2,3,4])
squared = nums.map(lambda x: x*x).collect()
for num in squared:
	print "%i " % (num)

# 对每个输入元素生成多个输出元素可以使用flatMap()， 一个返回值序列的迭代器
lines = sc.parallelize(["hello world", "hi"])
words = lines.flatMap(lambda line: line.split(" "))
words.first()

# aggregate()计算RDD平均值，可以替代map()后面接fold()的方式
sumCount = nums.aggregate((0,0),
	(lambda acc, value:(acc[0] + value, acc[1] + 1),
		(lambda acc1, acc2: (acc1[0] + acc2[0], acc1[1] + acc2[1]))))

#--------------------------------------------------------------------------#
# 在python中使用第一个单词作为键创建出一个pair RDD
pairs = lines.map(lambda x: (x.split(" ")[0], x))

# 对第二个元素进行筛选
result = pairs.filter(lambda keyValue: len(keyValue[1]) < 20)

# 使用reduceByKey()和mapValues()计算每个键对应的平均值
rdd.mapValues(lambda x: (x, 1)).reduceByKey(lambda x, y:(x[0] + y[0], x[1] +y[1]))

# 单词计数
rdd = sc.textFile("s3://...")
words = rdd.flatMap(lambda x: x.split(" "))
result = words.map(lambda x: (x, 1)).reduceByKey(lambda x, y: x + y)

# combineByKey()有多个参数分别对应聚合阶段的各个操作，适合用来解释聚合操作各个阶段的功能划分
# 求每个键对应的平均值
sumCount = nums.combineByKey((lambda x: (x, 1)),
	(lambda x, y: (x[0] + y, x[1] + 1)),
	(lambda x, y: (x[0] + y[0], x[1] + y[1]))
sumCount.map(lambda key, xy: (key, xy[0]/xy[1])).collectAsMap()

# 自定义reduceByKey()的并行度
data = [("a", 3), ("b", 4), ("a", 1)]
sc.parallelize(data).reduceByKey(lambda x, y: x + y)  # 默认并行度
sc.parallelize(data).reduceByKey(lambda x, y: x + y, 10)  # 自定义并行度

# 以字符串顺序整数进行自定义排序
rdd.sortByKey(ascending = True, numPartitions = None, keyfunc = lambda x: str(x))

# 自定义分区方式
import urlparse

def hash_domin(url):
	return hash(urlparse.urlparse(url).netloc)
rdd.partitionBy(20, hash_domin)  # 创建20个分区

#------------------------------------------------------------------------#
# 读取文本文件
input = sc.textFile("file:///home/holden/repos/spark/README.md")

# 将数据保存为文本文件
result.saveAsTextFile(ouputFile)


# 读取JSON
import json
data = inpur.map(lambda x: json.loads(x))

# 保存为JSON
(data.filter(lambda x: x["lovesPanas"]).map(lambda x: json.dumps(x)).saveAsTextFile(ouputFile))


# 使用textFile()读取CSV
import csv
import StringIO

def loadRecord(line):
	"""解析一行CSV记录"""
	input = StringIO.StringIO(line)
	reader = csv.DictReader(input, fieldnames = ["name", "favouriteAnimal"])
	return reader.next()

input = sc.textFile(inputFile).map(loadRecord)

# 完整读取CSV
def loadRecord(fileNameContents):
	"""读取给定文件中的所有记录"""
	input = StringIO.StringIO(fileNameContents[1])
	reader = csv.DictReader(input, fieldnames = ["name", "favouriteAnimal"])
	return reader

fullFileData = sc.wholeTextFiles(inputFile).flatMap(loadRecord)


#  写CSV
def writeRecords(records):
	"""写一些CSV记录"""
	output = StringIO.StringIO()
	writer = csv.DictWriter(output, fieldnames = ["name", "favouriteAnimal"])
	for record in records:
		writer.writerow(record)
	return [output.getvalue()]

pandasLovers.mapPartitions(writeRecords).saveAsTextFile(outputFile)


#  读取SequenceFile
val data = sc.sequenceFile(inFile,
	"org.apache.hadoop.io.Text", "org.apache.hadoop.io.IntWritable")


# 创建HiveContext并查询数据
from pyspark.sql import HiveContext

hiveCtx = HiveContext(sc)
rows = hiveCtx.sql("SELECT name, age FROM usrs")
firstRow = rows.first()
print firstRow.name

# 使用Spark SQL 读取 JSON 数据
tweets = hiveCtx.jsonFile("tweets.json")
tweets.registerTempTable("tweets")
results = hiveCtx.sql("SELECT usr.name, text FROM tweets")


#---------------------------------------------------------------------------#
# 呼叫日志示例：
"""
{"address":"address here", "band":"40m","callsign":"KK6JLK","city":"SUNNYVALE",
"contactlat":"37.384733","contactlong":"-122.032164",
"county":"Santa Clara","dxcc":"291","fullname":"MATTHEW McPherrin",
"id":57779,"mode":"FM","mylat":"37.751952821","mylong":"-122.4208688735",...}
"""

# 累加器 accumulator              对信息进行聚合
# 常见用途是在调试时对作业执行过程中的时间进行计数

累加器的用法如下所示。
• 通过在驱动器中调用 SparkContext.accumulator(initialValue) 方法，创建出存有初
始值的累加器。 返回值为 org.apache.spark.Accumulator[T] 对象，其中 T 是初始值
initialValue 的类型。
• Spark 闭包里的执行器代码可以使用累加器的 += 方法（在 Java 中是 add）增加累加器的值。
• 驱动器程序可以调用累加器的 value 属性（在 Java 中使用 value() 或 setValue()）来访
问累加器的值。

# 工作节点上的任务不能访问累加器的值

# 累加空行
file = sc.textFile(inputFile)

# c创建Accumulator[Int]并初始化为0
blankLines = sc.accumulator(0)

def extractCallSigns(line):
	global blankLines  # 访问全局变量
	if (line == ""):
		blankLines += 1
	return line.split(" ")

callSigns = file.flatMap(extractCallSigns)
callSigns.saveAsTextFile(outputDir + "/callsigns")
print "blank Lines: %d" % blankLines.value

# 使用累加器进行错误计数
validSignCount = sc.accumulator(0)
invalidSignCount = sc.accumulator(0)

def validateSign(sign):
	global validSignCount, invalidSignCount
	if re.match(r"\A\d?[a-zA-Z]{1,2}\d{1,4}[a-zA-Z]{1,3}\Z", sign):
		validSignCount += 1
		return True
	else:
		invalidSignCount += 1
		return False

# 对于每个呼号的联系次数进行计数
validSigns = callSigns.filter(validateSign)
contactCount = validSigns.map(lambda sign: (sign, 1)).reduceByKey(lambda (x, y): x + y)

# 强制求值计算计数
contactCount.count()
if invalidSignCount.value < 0.1 * validSignCount.value:
	contactCount.saveAsTextFile(outputDir + "/contactCount")
else:
	print "Too many errors: %d in %d" % (invalidSignCount.value, validSignCount.value)





# 广播变量 broadcast variable     高效分发较大的对象
# 可以让程序高效的向所有的工作节点发送一个较大的只读值，以供一个或多个spark操作使用。

# s使用广播变量查询国家
# 查询RDD contactCount 中的呼号的对应位置，将呼号前缀
# 读取为国家代码来进行查询
 signPrefixes = sc.broadcast(loadCallSignTable())

 def processSignCount(sign_count, signPrefixes):
 	country = lookupCountry(sign_count[0], signPrefixes)
 	count = sign_count[1]
 	return (country, count)

 countryContactCounts = (contactCounts
 	.map(processSignCount)
 	.reduceByKey((lambda x, y: x + y)))

 countryContactCounts.saveAsTextFile(outputDir + "/countries.txt")

 使用广播变量的过程很简单。
(1) 通过对一个类型 T 的对象调用 SparkContext.broadcast 创建出一个 Broadcast[T] 对象。
任何可序列化的类型都可以这么实现。
(2) 通过 value 属性访问该对象的值（在 Java 中为 value() 方法）。
(3) 变量只会被发到各个节点一次，应作为只读值处理（修改这个值不会影响到别的节点）。



#  使用共享连接池
def processCallSigns(signs):
	"""使用连接池查询呼号"""
	# 创建一个连接池
	http = urllib3.PoolManager()
	# 与每条呼号记录相关联的URL
	urls = map(lambda x: "http://73s.com/qsos/%s.json" % x, signs)
	# 创建请求（非阻塞）
	requests = map(lambda x: (x, http.request('GET', x)), urls)
	# 获取结果
	result = map(lambda x: (x[0], json.loads(x[1].data)). requests)
	# 删除空的结果并返回
	result = map(lambda x: x[1] is not None, result)

def fetchCallSigns(input):
	""" 获取呼号"""
	return input.mapPartitions(lambda callSigns: processSignCount(callSigns))

contactCountLsit = fetchCallSigns(validateSigns)


# 不使用mapPartitions()求平均值
def combineCtrs(c1, c2):
	return (c1[0] + c2[0], c[1] + c2[1])

def basicAvg(nums):
	"""计算平均值"""
	nums.map(lambda num: (num, 1)).reduce(combineCtrs)

# 使用。。。求平均值
def partitions(nums):
	""" 计算分区的sumCounter"""
	sumCount = [0,0]
	for num in nums:
		sumCount[0] += num
		sumCount[1] += 1
	return [sumCount]

def fastAvg(nums):
	"""计算平均值"""
	sumCount = nums.mapPartitions(partitionCtr).reduce(combineCtrs)
	return sumCount[0] / float(sumCount[1])



# 移除异常值
# 要把String类型RDD转为数字数据，这样才能使用统计函数并移除异常值
distanceNumerics = distances.map(lambda string: float(string))
stats = distancesNumerics.stats()
stddev = std.stddev()
mean = stats.mean()
resonableDistances = distancesNumerics.filter(
	lambda x: math.fabs(x - mean) < 3 * stddev)
print reasonableDistances.collect()

# --------------------------------------------------------------------#

跳过在集群上运行Spark的相关内容

#-------------------------------------------------------------------- #

Spark 驱动器是执行你的程序中的 main() 方法的进程。它执行用户编写的用来创建
SparkContext、创建 RDD，以及进行 RDD 的转化操作和行动操作的代码。

驱动器程序在 Spark 应用中有下述两个职责。
• 把用户程序转为任务
Spark 驱动器程序负责把用户程序转为多个物理执行的单元， 这些单元也被称为任务
（ task）。从上层来看，所有的 Spark 程序都遵循同样的结构：程序从输入数据创建一系
列 RDD， 再使用转化操作派生出新的 RDD，最后使用行动操作收集或存储结果 RDD
中的数据。 Spark 程序其实是隐式地创建出了一个由操作组成的逻辑上的有向无环图
（ Directed Acyclic Graph，简称 DAG）。当驱动器程序运行时，它会把这个逻辑图转为物
理执行计划。
Spark 会对逻辑执行计划作一些优化，比如将连续的映射转为流水线化执行，将多个操
作合并到一个步骤中等。 这样 Spark 就把逻辑计划转为一系列步骤（ stage）。而每个步
骤又由多个任务组成。这些任务会被打包并送到集群中。任务是 Spark 中最小的工作单
元，用户程序通常要启动成百上千的独立任务。

• 为执行器节点调度任务
有了物理执行计划之后， Spark 驱动器程序必须在各执行器进程间协调任务的调度。执行
器进程启动后，会向驱动器进程注册自己。因此，驱动器进程始终对应用中所有的执行
器节点有完整的记录。每个执行器节点代表一个能够处理任务和存储 RDD 数据的进程。
Spark 驱动器程序会根据当前的执行器节点集合， 尝试把所有任务基于数据所在位置分
配给合适的执行器进程。 当任务执行时，执行器进程会把缓存数据存储起来，而驱动器
进程同样会跟踪这些缓存数据的位置， 并且利用这些位置信息来调度以后的任务，以尽
量减少数据的网络传输。

# 提交python应用
bin/spark-submit my_script.py

#-------------------------------------------------------------------------#

# 对Spark进行行嗯那个调优，通常就是修改spark应用的运行时配置选项。
# spark中最主要的配置机制是通过SparkConf类对spark进行配置
# 当创建出一个SparkContext时，就需要创建出一个SparkConf 的实例
 
# 使用SparkConf创建一个应用
# 创建一个conf对象
conf = new SparkConf()
conf.set("spark.app.name", "My Spark App")
conf.set("spark.master", "local[4]")
conf.set("speak.ui.port", "36000")  # 重载默认端口配置

# 使用这个配置对象创建一个SparkContext
sc = SparkContext(conf)


# 特定的行动操作所生成的不周到额集合称为一个作业

# 深度调优阅读官方指南 http://spark.apache.org/docs/latest/tuning.html

#-------------------------------------------------------------------------------#

# python中SQL的import 声明
# 导入Spark SQL
from pyspark.sql import HiveContext, Row
# 当不能引入hive依赖时
from pyspark.sql import SQLContext, Row

# 添加好import 声明后，需要创建出一个HiveContext 对象
# 在python中创建SQL上下文环境
hiveCtx = HiveContext(sc)

# 基本查询示例
input = hiveCtx.jsonFile(inputFile)
# 注册输入的SchemaRDD
input.registerTempTable("tweets")
# 依赖retweetCount（转发计数）选出推文
topTweets = hiveCtx.sql(""" SELECT text, retweetCount FROM
tweets ORDER BY retweetCount LIMIT 10""")


# 在python中访问topTweet这个SchemaRDD中的text列
topTweetText = topTweets.map(lambda row: row.text)

# 使用 python从Hive 读取
from pyspark.sql import HiveContext

hiveCtx = HiveContext(sc)
rows = hiveCtx.sql("SELECT key，value FROM mytable")
keys = rows.map(lambda row: row[0])

# python中的Parquet数据读取
# 从一个有name和favouriteAnimal字段的Parquet文件中读取数据
rows = hiveCtx.parquetFile(parquetFile)
names = rows.map(lambda row:row.name)
print "Everyone"
print names.collect()

# python 中的Parqust数据查询
# 寻找熊猫爱好者
tbl = rows.registerTempTable("people")
pandaFriends = hiveCtx.sql("SELECT name FROM people WHERE favouriteAnimal = \"panda\"")
print "Panda friends"
print pandaFriends.map(lambda row: row.name).collect()

# Parquet文件保存
pandaFriends.saveAsTextFile("hdfs://...")


# 在python中使用Spark SQL 读取JSON数据
input = hiveCtx.jsonFile(inputFile)

#  在python中使用Row和具名远足创建SchemaRDD
happyPeopleRDD = sc.parallelize([Row(name = "holden", favouriteBeverage = "coffee")])
happyPeopleSchemaRDD = hiveCtx.inferSchema(happyPeopleRDD)
happyPeopleSchemaRDD.registerTempTable("happy_people")

#****************  使用Beeline    JDBC/ODBC服务器   ************************************

# 用户自定义函数

# 字符串长度 UDF
# 写一个求字符串长度的UDF
hiveCtx.registerFunction("strLenPython", lambda x: len(x), IntegerType())
lengthSchemaRDD = hiveCtx.sql("SELECT strLenPython('text') FROM tweets LIMIT 10")


# ----------------------------------------------------------------------------------########

Spark Streaming使用离散化流(discretized stream)作为抽象表示，叫做DStream。

#***********************   Streaming ***************************************************
