from pyspark.sql import SparkSession
import pyspark.sql.functions as F

spark = SparkSession.builder\
        .master("local[2]")\
        .appName("Question2")\
        .config("spark.local.dir", "/mnt/parscratch/users/acp23ws")\
        .getOrCreate()

sc = spark.sparkContext
sc.setLogLevel("WARN")

# load data csv
train_raw_DF = spark.read.csv('./Data/5xor_128bit/train_5xor_128dim.csv')
test_raw_DF = spark.read.csv('./Data/5xor_128bit/test_5xor_128dim.csv')

rand_seed = 66896

# A. Small dataset
# random select 1% of original data
small_train_DF = spark.createDataFrame(train_raw_DF).sample(fraction=0.01, seed=rand_seed)
small_test_DF = spark.createDataFrame(test_raw_DF).sample(fraction=0.01, seed=rand_seed)
print("small dataset count:")
print(small_train_DF.count())
print(small_test_DF.count())

### class balancing?

small_train_DF.printSchema()

# pipeline
# Random Forests

# Gradient boosting 
# and (shallow) Neural networks
