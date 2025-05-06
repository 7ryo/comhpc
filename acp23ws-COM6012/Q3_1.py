from pyspark.sql import SparkSession
import pyspark.sql.functions as F

spark = SparkSession.builder\
        .master("local[2]")\
        .appName("Question2")\
        .config("spark.local.dir", "/mnt/parscratch/users/acp23ws")\
        .getOrCreate()

sc = spark.sparkContext
sc.setLogLevel("ERROR")

# load data csv
train_raw_DF = spark.read.csv('./Data/5xor_128bit/train_5xor_128dim.csv')
test_raw_DF = spark.read.csv('./Data/5xor_128bit/test_5xor_128dim.csv')

rand_seed = 66896

# A. Small dataset
# random select 1% of original data
small_train_DF = train_raw_DF.sample(fraction=0.01, seed=rand_seed)
small_test_DF = test_raw_DF.sample(fraction=0.01, seed=rand_seed)
print("small dataset count:")
print(small_train_DF.count())
print(small_test_DF.count())

### class balancing?

#small_train_DF.printSchema()

small_train_DF = small_test_DF.withColumn('labels', F.when(F.col('_c128')==-1, 0.0).otherwise(1.0)).drop('_c128')
small_test_DF = small_test_DF.withColumn('labels', F.when(F.col('_c128')==-1, 0.0).otherwise(1.0)).drop('_c128')

# value types are string
# convert them into double/int
from pyspark.sql.types import StringType
StringColumns = [x.name for x in small_train_DF.schema.fields if x.dataType == StringType()]
for c in StringColumns:
    small_train_DF = small_train_DF.withColumn(c, F.col(c).cast("double"))
    small_test_DF = small_test_DF.withColumn(c, F.col(c).cast("double"))

#small_train_DF.printSchema() # they are double now

# vector assembler to concat all features -> one vector
from pyspark.ml.feature import VectorAssembler
feature_names = small_train_DF.columns
num_features = len(feature_names)
vecAssembler = VectorAssembler(inputCols = feature_names[:-1], outputCol = 'features') 
testvecTrainingData = vecAssembler.transform(small_train_DF)
testvecTrainingData.show(5)

# pipeline
from pyspark.ml import Pipeline
from pyspark.ml.regression import RandomForestRegressor

rf = RandomForestRegressor(labelCol="labels", featuresCol="features", maxDepth=5, numTrees=3, \
                           featureSubsetStrategy = 'all', seed=123, bootstrap=False)
# Random Forests

stages = [vecAssembler, rf]
pipeline = Pipeline(stages=stages)

pipelineModel = pipeline.fit(small_train_DF)
predictions = pipelineModel.transform(small_test_DF)

predictions.select('features', 'labels', 'pediction').show(10)

# Eval
# classification accuracy
from pyspark.ml.evaluation import MulticlassClassificationEvaluator 
eval_multi = MulticlassClassificationEvaluator(labelCol="labels", predictionCol="prediction", metricName="accuracy")
accuracy = eval_multi.evaluate(predictions)
print("Accuracy = %g " % accuracy)

# area under the curve
from pyspark.ml.evaluation import BinaryClassificationEvaluator

# Gradient boosting 
# and (shallow) Neural networks
# and (shallow) Neural networks
