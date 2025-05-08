##### ALS
from pyspark.sql import SparkSession, Row
spark = SparkSession.builder\
        .master("local[2]")\
        .appName("Question2")\
        .config("spark.local.dir", "/mnt/parscratch/users/acp23ws")\
        .getOrCreate()

sc = spark.sparkContext
sc.setLogLevel("ERROR")
# A. Recommender w/ ALS
# four-fold cross-validation ALS-based recommendation 
# ==================================================== #
# data: ratings.csv
#       split into 4 folds
#       train:test = 75%:25%
#       repeat four times (manually)
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml.recommendation import ALS
###from pyspark.sql import Row
import pyspark.sql.functions as F

# load data from csv
raw_rating_DF = spark.read.csv('./Data/ml-25m/ratings.csv', header=True).cache()
raw_rating_DF.show(5)
print(f"there are {raw_rating_DF.count()} rows of data in ratings.csv\n")
# check data type
raw_rating_DF.printSchema()

# pre processing
rand_seed = 1840977
splits = raw_rating_DF.randomSplit([0.25, 0.25, 0.25, 0.25], seed=rand_seed)
## use union to combine splits?
## or use subtract to remove current split?
training = splits[1].join(splits[2]).join(splits[3]).cache()
test = splits[0].cache()
print(f"training count = {training.count()}")
print(f"test count = {test.count()}")


# 3 versions of ALS
# setting1) use the settings in Lab8
#           change random seed = student number
als_1 = ALS(userCol="uesrId", itemCol="movieId", \
		seed=rand_seed, coldStartStrategy="drop")
model_1 = als.fit(training)
predictions = model_1.transform(test)
evaluator = RegressionEvaluator(metricName="rmse", labelCol="rating", predictionCol="prediction")
rmse = evaluator.evaluate(predictions)
print(f"RMSE = {rmse}")
# setting2)
# setting3)
# eg. changing rank, regParam, alpha
# improve the model

# Eval: 
# for each split - mean RMSE and mean MAE for 3 ALSs
# over four splits -
# [mean&standard deviation] of RMSE and MAE
# report => put all 36 numbers in a table

# plot => mean&std of RMSE and MAE for each of 3ver ALS

# ===================================================== #
# B. K-Means
