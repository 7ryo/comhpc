from pyspark.sql import SparkSession
from pyspark.sql.types import StringType
import pyspark.sql.functions as F
import numpy as np

spark = SparkSession.builder\
        .master("local[2]")\
        .appName("Question2")\
        .getOrCreate()

sc = spark.sparkContext
sc.setLogLevel("ERROR")

# load data csv -. spark dataframe
train_raw_DF = spark.read.csv('acp23ws-COM6012/Data/5xor_128bit/train_5xor_128dim.csv')
test_raw_DF = spark.read.csv('acp23ws-COM6012/Data/5xor_128bit/test_5xor_128dim.csv')

rand_seed = 66896

# A. Small dataset
# random select 1% of original data
small_train_DF = train_raw_DF.sample(fraction=0.01, seed=rand_seed)
small_test_DF = test_raw_DF.sample(fraction=0.01, seed=rand_seed)
print("small dataset count:")
print(small_train_DF.count())
print(small_test_DF.count())

### class balancing?

# convert label (1, -1) inot [0,100]
# Labels MUST be in [0, 100], but got -1.0
small_train_DF = small_test_DF.withColumn('labels', F.when(F.col('_c128')==-1, 0.0).otherwise(1.0)).drop('_c128')


# convert value from stringType to Double
StringColumns = [x.name for x in small_train_DF.schema.fields if x.dataType == StringType()]
for c in StringColumns:
    small_train_DF = small_train_DF.withColumn(c, F.col(c).cast("double"))

# TODO: also test data
# StringColumns = [x.name for x in small_test_DF.schema.fields if x.dataType == StringType()]
# for c in StringColumns:
#     small_test_DF = small_test_DF.withColumn(c, F.col(c).cast("double"))

small_train_DF.printSchema()



# vectorassembler
feature_names = small_train_DF.columns
num_features = len(feature_names)

from pyspark.ml.feature import VectorAssembler
vecAssembler = VectorAssembler(inputCols = feature_names[0:num_features-1], outputCol = 'features') 
vecTrainingData = vecAssembler.transform(small_train_DF)
vecTrainingData.select("features", feature_names[-1]).show(5)

from pyspark.ml.classification import DecisionTreeClassifier
dt = DecisionTreeClassifier(labelCol="labels", featuresCol="features", maxDepth=10, impurity='entropy')
model = dt.fit(vecTrainingData)

fi = model.featureImportances
imp_feat = np.zeros(num_features-1)
imp_feat[fi.indices] = fi.values

import matplotlib 
matplotlib.use('Agg')

import matplotlib.pyplot as plt
x = np.arange(num_features-1)
plt.bar(x, imp_feat)
plt.savefig("feature_importances.png")
