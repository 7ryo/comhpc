from pyspark.sql import SparkSession
from pyspark.sql.functions import regexp_extract, col, when, rank
from pyspark.sql.window import Window

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
            .withColumn('bytes in the reply', regexp_extract('value', pattern, 5))\
            .withColumn('country', when(col('host').endswith('.ac.uk'), "UK")\
                                    .when(col('host').endswith('.edu'), "US")\
                                    .when(col('host').contains('.edu.au'), "Australia")\
                                    .otherwise("other"))\
            .drop('value').cache()

df.show(10, False)

# count total number of unique instituitions
num_host = df.select('host').distinct().count()
print("total number of distinct hosts is %i.\n" % num_host)


# top 9 most frequent visitors (9 per country)
#hostcount_uk = df.select('host').groupBy('')
counted_host = df.select('country', 'host').groupBy('country', 'host').count().sort('count', ascending=False).cache()

print("Count of all hosts")
counted_host.show(10, False)

print("Most frequent 9 hosts per country")
# counted_host.filter(col('host').contains('.ac.uk')).show(10, False)
# counted_host.filter(col('host').endswith('.edu')).show(10, False)
# counted_host.filter(col('host').contains('.edu.au')).show(10, False)
window_spec = Window.partitionBy('country').orderBy(col('count').desc())
ranked_host = counted_host.withColumn("rank", rank().over(window_spec)).cache()
ranked_host.show(10, False)

top9_host = ranked_host.filter(col('rank') <= 9).orderBy('country', 'rank').cache()
top9_host.show(27, False)



# find out the ranking of Univerisity of Sheffield (UK)
# .shef.ac.uk
ranked_host.filter(col('host').contains('.shef.ac.uk')).show(30)

spark.stop()
