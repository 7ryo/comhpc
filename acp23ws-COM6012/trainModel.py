from pyspark.sql import SparkSession
import pyspark.sql.functions as F
from pyspark.sql.types import StringType
import argparse
import time
from pyspark.ml import Pipeline
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.classification import RandomForestClassifier, GBTClassifier, MultilayerPerceptronClassifier
from pyspark.ml.evaluation import MulticlassClassificationEvaluator, BinaryClassificationEvaluator
import pandas as pd

spark = SparkSession.builder\
        .master("local[2]")\
        .appName("Question3")\
        .config("spark.local.dir", "/mnt/parscratch/users/acp23ws")\
        .getOrCreate()

sc = spark.sparkContext
sc.setLogLevel("ERROR")

# get params from command line
parser = argparse.ArgumentParser()
parser.add_argument('--model', type=str, required=True)
parser.add_argument('--datasize', type=float, required=True)
args = parser.parse_args()

# time
start_time = time.time()

# load whole data csv
train_raw_DF = spark.read.csv('./Data/5xor_128bit/train_5xor_128dim.csv')
test_raw_DF = spark.read.csv('./Data/5xor_128bit/test_5xor_128dim.csv')

rand_seed = 66896

# get the size
small_train_DF = train_raw_DF.sample(fraction=args.datasize, seed=rand_seed)
small_test_DF = test_raw_DF.sample(fraction=args.datasize, seed=rand_seed)
# label column
small_train_DF = small_test_DF.withColumn('labels', F.when(F.col('_c128')==-1, 0.0).otherwise(1.0)).drop('_c128')
small_test_DF = small_test_DF.withColumn('labels', F.when(F.col('_c128')==-1, 0.0).otherwise(1.0)).drop('_c128')
# feature value types are string
# convert them into double
StringColumns = [x.name for x in small_train_DF.schema.fields if x.dataType == StringType()]
for c in StringColumns:
    small_train_DF = small_train_DF.withColumn(c, F.col(c).cast("double"))
    small_test_DF = small_test_DF.withColumn(c, F.col(c).cast("double"))

# define assembler
feature_names = small_train_DF.columns
num_features = len(feature_names)
vecAssembler = VectorAssembler(inputCols = feature_names[:-1], outputCol = 'features') 
# define evaluator
eval_acc = MulticlassClassificationEvaluator(labelCol="labels", predictionCol="prediction", metricName="accuracy")
eval_auc = BinaryClassificationEvaluator(rawPredictionCol="rawPrediction", labelCol="labels", metricName="areaUnderROC")

# define models
model = {
    'RF': RandomForestClassifier(featuresCol="features", labelCol="labels",\
                                 maxDepth=7, maxBins=16, numTrees=5,\
				                 seed=rand_seed),
    'GBT': GBTClassifier(featuresCol="features", labelCol="labels", 
                         maxDepth=5, maxBins=16,maxIter=5, \
                         seed=rand_seed),
    'MPC': MultilayerPerceptronClassifier(featuresCol="features", labelCol="labels", \
                                          blockSize=32, layers=[128, 15, 8, 2], maxIter=50,\
                                          seed=rand_seed)
}

pipeline = Pipeline(stages=[vecAssembler, model[args.model]])
pipelineModel = pipeline.fit(small_train_DF)
predictions = pipelineModel.transform(small_test_DF)
acc = eval_acc.evaluate(predictions)
auc = eval_auc.evaluate(predictions)

end_time = time.time()
runtime = end_time - start_time

# update the .csv
runtime_DF = pd.read_csv('./log_runtime.csv', index_col=0)

cond = (runtime_DF['model']==args.model) & (runtime_DF['datasize']==args.datasize)
runtime_DF.loc[cond, 'runtime'] = runtime
runtime_DF.loc[cond, 'accuracy'] = acc
runtime_DF.loc[cond, 'auc'] = auc

runtime_DF.to_csv('./log_runtime.csv', header=True)
