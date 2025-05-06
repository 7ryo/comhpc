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
###testvecTrainingData = vecAssembler.transform(small_train_DF)
###testvecTrainingData.show(5)

# =================================================== #
# 1. use pipeline + crossval train models
from pyspark.ml import Pipeline
from pyspark.ml.tuning import CrossValidator

# 1) Random Forests
from pyspark.ml.classification import RandomForestClassifier
## use classifier, not regressor ##
rf = RandomForestClassifier(featuresCol="features", labelCol="labels", maxDepth=5, numTrees=3,\
				seed=rand_seed)
pipeline_rf = Pipeline(stages=[vecAssembler, rf])

pipelineModel_rf = pipeline_rf.fit(small_train_DF)
predictions_rf = pipelineModel_rf.transform(small_test_DF)
##predictions.select('features', 'labels', 'prediction').show(10)

# 2) Gradient boosting 
from pyspark.ml.classification import GBTClassifier
gbt = GBTClassifier(featuresCol="features", labelCol="labels", maxDepth=5, maxIter=5,\
                    seed=rand_seed)
pipeline_gbt = Pipeline(stages=[vecAssembler, gbt])
pipelineModel_gbt = pipeline_gbt.fit(small_train_DF)
predictions_gbt = pipelineModel_gbt.transform(small_test_DF)


# 3) (shallow) Neural networks
from pyspark.ml.classification import MultilayerPerceptronClassifier
### input = 23 features?
layers = [num_features-1, 10, 3]
mpc = MultilayerPerceptronClassifier(featuresCol="features", labelCol="labels", maxIter=50,\
                                     layers=layers, seed=rand_seed)
pipeline_mpc = Pipeline(stages=[vecAssembler, mpc])
pipelineModel_mpc = pipeline_mpc.fit(small_train_DF)
predictions_mpc = pipelineModel_mpc.transform(small_test_DF)

# ==================================================== #
# 2. Eval
# classification accuracy
from pyspark.ml.evaluation import MulticlassClassificationEvaluator 
eval_multi = MulticlassClassificationEvaluator(labelCol="labels", predictionCol="prediction", metricName="accuracy")
accuracy_rf = eval_multi.evaluate(predictions_rf)
print(f"Accuracy of rf = {accuracy_rf}")
accuracy_gbt = eval_multi.evaluate(predictions_gbt)
print(f"Accuracy of rf = {accuracy_gbt}")
accuracy_mpc = eval_multi.evaluate(predictions_rf)
print(f"Accuracy of rf = {accuracy_mpc}")

# area under the curve
from pyspark.ml.evaluation import BinaryClassificationEvaluator
