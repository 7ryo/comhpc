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
raw_rating_DF = raw_rating_DF.withColumn('userId', F.col('userId').cast("long"))
raw_rating_DF = raw_rating_DF.withColumn('movieId', F.col('movieId').cast("long"))
raw_rating_DF = raw_rating_DF.withColumn('rating', F.col('rating').cast("double"))

# check data type
raw_rating_DF.printSchema()

rand_seed = 1840977

# reduce the size
#raw_rating_DF = raw_rating_DF.sample(fraction=0.8, seed=rand_seed)

# pre processing
# rand_seed = 1840977
splits = raw_rating_DF.randomSplit([0.25, 0.25, 0.25, 0.25], seed=rand_seed)
## use union to combine splits?
## or use subtract to remove current split?

# print(f"training count = {training.count()}")
# print(f"test count = {test.count()}")


# 3 versions of ALS
evaluator_RMSE = RegressionEvaluator(metricName="rmse", labelCol="rating", predictionCol="prediction")
evaluator_MAE = RegressionEvaluator(metricName="mae", labelCol="rating", predictionCol="prediction")

# setting1) use the settings in Lab8
#           change random seed = student number
als_1 = ALS(userCol="userId", itemCol="movieId", \
		seed=rand_seed, coldStartStrategy="drop")
# model_1 = als_1.fit(training)
# predictions = model_1.transform(test)
# rmse = evaluator.evaluate(predictions)
# print(f"RMSE = {rmse}")
# setting2)
als_2 = ALS(userCol="userId", itemCol="movieId", \
		seed=rand_seed, coldStartStrategy="drop",\
                rank=20, blockSize=1024, maxIter=15)
# setting3)
als_3 = ALS(userCol="userId", itemCol="movieId", \
		seed=rand_seed, coldStartStrategy="drop",
                rank=40, blockSize=1024, regParam=0.05)
# eg. changing rank, regParam, alpha
# improve the model
from pyspark.sql.types import StructType, StructField, StringType, DoubleType

schema = StructType([
    StructField("no_split", StringType(), True),
    StructField("model1_RMSE", DoubleType(), True),
    StructField("model1_MAE", DoubleType(), True),
    StructField("model2_RMSE", DoubleType(), True),
    StructField("model2_MAE", DoubleType(), True),
    StructField("model3_RMSE", DoubleType(), True),
    StructField("model3_MAE", DoubleType(), True),
])

#evals_DF = spark.createDataFrame([], schema)
#evals_DF = spark.createDataFrame([], ['no_split','model1_RMSE', 'model1_MAE', 'model2_RMSE', 'model2_MAE', 'model3_RMSE', 'model3_MAE'])
import pandas as pd
evals_DF = pd.DataFrame([[0,0,0,0,0,0]], columns=['rmse_1', 'mae_1', 'rmse_2', 'mae_2', 'rmse_3', 'mae_3'])

for i in range(4):
    print(f"i={i}")
    test = splits[i].cache()
    training = raw_rating_DF.subtract(test).cache()
    model_1 = als_1.fit(training)
    predictions = model_1.transform(test)
    rmse_1 = evaluator_RMSE.evaluate(predictions)
    mae_1 = evaluator_MAE.evaluate(predictions)

    model_2 = als_2.fit(training)
    predictions = model_2.transform(test)
    rmse_2 = evaluator_RMSE.evaluate(predictions)
    mae_2 = evaluator_MAE.evaluate(predictions)

    model_3 = als_3.fit(training)
    predictions = model_3.transform(test)
    rmse_3 = evaluator_RMSE.evaluate(predictions)
    mae_3 = evaluator_MAE.evaluate(predictions)

    # free cache
    test.unpersist()
    training.unpersist()

    evals_DF.loc[i] = [rmse_1, mae_1, rmse_2, mae_2, rmse_3, mae_3]
evals_DF.to_csv('log_evals_3.csv')

# Eval: 
# for each split - mean RMSE and mean MAE for 3 ALSs
# over four splits -
# [mean&standard deviation] of RMSE and MAE
# report => put all 36 numbers in a table
#log_eval_DF = spark.read.csv('log_evals.csv', header=True)
#log_eval_DF = log_eval_DF.withColumn('mean_rmse', (F.col('rmse_1')+F.col('rmse_2')+F.col('rmse_3'))/3)\
#                        .withColumn('mean_mae', (F.col('mae_1')+F.col('mae_2')+F.col('mae_3'))/3)
#log_eval_DF.show()

#df = log_eval_DF.select(
#    F.mean("mean_rmse").alias("mean_mean_rmse"),
#    F.stddev("mean_rmse").alias("std_mean_rmse"),
#    F.mean("mean_mae").alias("mean_mean_mae"),
#    F.stddev("mean_mae").alias("std_mean_mae")
#)
#df.show()
                        
# plot => mean&std of RMSE and MAE for each of 3ver ALS

# ===================================================== #
# B. K-Means
