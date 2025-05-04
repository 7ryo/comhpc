from pyspark.sql import SparkSession
from pyspark.sql.functions import regexp_extract, count_distinct

spark = SparkSession.builder\
        .master("local[2]")\
        .appName("Question1")\
        .config("spark.local.dir", "/mnt/parscratch/users/acp23ws")\
        .getOrCreate()

sc = spark.sparkContext
sc.setLogLevel("WARN")

logFile = spark.read.text("./Data/NASA_access_log_Jul95.gz").cache()

# convert logFile into DataFrame based on the five columns

# example log:
# in24.inetnebr.com - - [01/Aug/1995:00:00:01 -0400] "GET /shuttle/missions/sts-68/news/sts-68-mcc-05.txt HTTP/1.0" 200 1839
# in the order of host, timestamp, request, http reply code, bytes in the reply

# use regex to extract each part
# 

pattern = r'^(.*?) - - \[(.*?)\] "(.*?)" (\d+) (\d+)'

df = logFile.withColumn('host', regexp_extract('value', pattern, 1))\
            .withColumn('timestamp', regexp_extract('value', pattern, 2))\
            .withColumn('request', regexp_extract('value', pattern, 3))\
            .withColumn('HTTP reply code', regexp_extract('value', pattern, 4))\
            .withColumn('bytes in the reply', regexp_extract('value', pattern, 5)).drop('value').cache()

df.show(10, False)

# count total number of unique instituitions
num_host = df.select('host').distinct().count()
print(num_host)
print(f"use countDistinct will be: {df.select(count_distinct('host'))}")


# top 9 most frequent visitors (9 per country)
#hostcount_uk = df.select('host').groupBy('')
hostcount = df.select('host').groupBy('host').count().sort('count', ascending=False)
hostcount.show(10, False)

# find out the ranking of Univerisity of Sheffield (UK)
# .shef.ac.uk