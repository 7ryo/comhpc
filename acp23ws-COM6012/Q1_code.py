#Question 1
from pyspark.sql import SparkSession

spark = SparkSession.builder \
	.master("local[2]") \
	.appName("Question1") \
	.config("spark.local.dir", "/mnt/parscratch/users/acp23ws") \
	.getOrCreate()

sc = spark.sparkContext
sc.setLogLevel("WARN")

# load file
# NASA_access_log_Jul95.gz
with open ('./Data/NASA_access_log_Jul95.gz', 'r') as f:
	file_content = f.read()
	print(file_content)
