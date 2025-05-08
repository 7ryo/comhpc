from pyspark.sql import SparkSession
import pyspark.sql.functions as F
from pyspark.sql.types import StringType

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
StringColumns = [x.name for x in small_train_DF.schema.fields if x.dataType == StringType()]
for c in StringColumns:
    small_train_DF = small_train_DF.withColumn(c, F.col(c).cast("double"))
    small_test_DF = small_test_DF.withColumn(c, F.col(c).cast("double"))


# =================================================== #
# 1. use pipeline + crossval train models
from pyspark.ml import Pipeline
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.tuning import CrossValidator, ParamGridBuilder
from pyspark.ml.classification import RandomForestClassifier, GBTClassifier, MultilayerPerceptronClassifier
from pyspark.ml.evaluation import MulticlassClassificationEvaluator 

# vector assembler: to concat all features -> one vector
feature_names = small_train_DF.columns
num_features = len(feature_names)
vecAssembler = VectorAssembler(inputCols = feature_names[:-1], outputCol = 'features') 

# evaluator:
# 1) accuracy
eval_acc = MulticlassClassificationEvaluator(labelCol="labels", predictionCol="prediction", metricName="accuracy")
# 2) auc

# 1) Random Forests
## use classifier, not regressor ##
rf = RandomForestClassifier(featuresCol="features", labelCol="labels",\
				            seed=rand_seed)
pipeline_rf = Pipeline(stages=[vecAssembler, rf])
paramGrid_rf = ParamGridBuilder()\
                .addGrid(rf.maxDepth, [3, 5, 7])\
                .addGrid(rf.maxBins, [16, 32, 64])\
                .addGrid(rf.numTrees, [5, 20, 50]) \
		.build()
crossval_rf = CrossValidator(estimator=pipeline_rf,
                             estimatorParamMaps=paramGrid_rf,
                             evaluator=eval_acc)
print("training rf")
#cvModel_rf_acc = crossval_rf.fit(small_train_DF)
#### select best model param
#bestmodel = cvModel_rf_acc.bestModel.stages[-1]
#best_rf_paramDict = {
#    'maxDepth': bestmodel.getOrDefault('maxDepth'),
#    'maxBins': bestmodel.getOrDefault('maxBins'),
#    'numTrees': bestmodel.getOrDefault('numTrees')
#}
#print(f"best param of rf (eval=acc) is {best_rf_paramDict}")

#prediction = cvModel_rf_acc.transform(small_test_DF)
#acc_rf = eval_acc.evaluate(prediction)

##no crossval##
#pipelineModel_rf = pipeline_rf.fit(small_train_DF)
#predictions_rf = pipelineModel_rf.transform(small_test_DF)
##predictions.select('features', 'labels', 'prediction').show(10)

# 2) Gradient boosting 
gbt = GBTClassifier(featuresCol="features", labelCol="labels",\
                    seed=rand_seed)
pipeline_gbt = Pipeline(stages=[vecAssembler, gbt])
paramGrid_gbt = ParamGridBuilder()\
                .addGrid(gbt.maxDepth, [3, 5, 7])\
                .addGrid(gbt.maxBins, [16, 32, 64])\
                .addGrid(gbt.maxIter, [3, 5, 7]) \
		.build()
print("training gbt")
crossval_gbt = CrossValidator(estimator=pipeline_gbt,
                             estimatorParamMaps=paramGrid_gbt,
                             evaluator=eval_acc)
cvModel_gbt_acc = crossval_gbt.fit(small_train_DF)
#### select best model param
bestmodel = cvModel_gbt_acc.bestModel.stages[-1]
best_gbt_paramDict = {
    'maxDepth': bestmodel.getOrDefault('maxDepth'),
    'maxBins': bestmodel.getOrDefault('maxBins'),
    'maxIter': bestmodel.getOrDefault('maxIter')
}
print(f"best param of gbt (eval=acc) is {best_gbt_paramDict}")

prediction = cvModel_gbt_acc.transform(small_test_DF)
acc_gbt = eval_acc.evaluate(prediction)


# pipelineModel_gbt = pipeline_gbt.fit(small_train_DF)
# predictions_gbt = pipelineModel_gbt.transform(small_test_DF)


# 3) (shallow) Neural networks
### input = 23 features?
layers = [[num_features-1, 10, 2],\
          [num_features-1, 15, 8, 2]] #binary classification -> output layer = 2
mpc = MultilayerPerceptronClassifier(featuresCol="features", labelCol="labels", \
                                     seed=rand_seed)
pipeline_mpc = Pipeline(stages=[vecAssembler, mpc])
print("training mpc")
paramGrid_mpc = ParamGridBuilder()\
                .addGrid(mpc.blockSize, [32, 64, 128])\
                .addGrid(mpc.layers, layers)\
                .addGrid(mpc.maxIter, [20, 50, 100])\
		.build()
crossval_mpc = CrossValidator(estimator=pipeline_mpc,
                             estimatorParamMaps=paramGrid_mpc,
                             evaluator=eval_acc)
#cvModel_mpc_acc = crossval_mpc.fit(small_train_DF)
#### select best model param
#bestmodel = cvModel_mpc_acc.bestModel.stages[-1]
#best_mpc_paramDict = {
#    'blockSize': bestmodel.getOrDefault('blockSize'),
#    'layers': bestmodel.getOrDefault('layers'),
#    'maxIter': bestmodel.getOrDefault('maxIter')
#}
#print(f"best param of mpc (eval=acc) is {best_mpc_paramDict}")
#prediction = cvModel_mpc_acc.transform(small_test_DF)
#acc_mpc = eval_acc.evaluate(prediction)

# pipelineModel_mpc = pipeline_mpc.fit(small_train_DF)
# predictions_mpc = pipelineModel_mpc.transform(small_test_DF)

# ==================================================== #
# 2. Eval w/ aUC
# classification accuracy


#accuracy_rf = eval_multi.evaluate(predictions_rf)
#print(f"Accuracy of rf using cvModel_ = {acc_rf}")
#accuracy_gbt = eval_multi.evaluate(predictions_gbt)
#print(f"Accuracy of gbt = {acc_gbt}")
#accuracy_mpc = eval_multi.evaluate(predictions_rf)
#print(f"Accuracy of mpc = {acc_mpc}")

# Eval: area under the curve
from pyspark.ml.evaluation import BinaryClassificationEvaluator
eval_auc = BinaryClassificationEvaluator(rawPredictionCol="rawPrediction", labelCol="labels", metricName="areaUnderROC")

## create another crossvalidator
#crossval_rf2 = CrossValidator(estimator=pipeline_rf,
#                              estimatorParamMaps=paramGrid_rf,
#                              evaluator=eval_auc)
#cvModel_rf_auc = crossval_rf2.fit(small_train_DF)
#### select best model param
#bestmodel = cvModel_rf_auc.bestModel.stages[-1]
#best_rf_paramDict = {
#    'maxDepth': bestmodel.getOrDefault('maxDepth'),
#    'maxBins': bestmodel.getOrDefault('maxBins'),
#    'numTrees': bestmodel.getOrDefault('numTrees')
#}
#print(f"best param of rf (eval=auc) is {best_rf_paramDict}") #TODO: compare?

#prediction = cvModel_rf_auc.transform(small_test_DF)
#auc_rf = eval_auc.evaluate(prediction)
#print(f"AUC of rf using cvModel_ = {auc_rf}")

# gbt
crossval_gbt2 = CrossValidator(estimator=pipeline_gbt,
                             estimatorParamMaps=paramGrid_gbt,
                             evaluator=eval_auc)
cvModel_gbt_auc = crossval_gbt2.fit(small_train_DF)
#### select best model param
bestmodel = cvModel_gbt_auc.bestModel.stages[-1]
best_gbt_paramDict = {
    'maxDepth': bestmodel.getOrDefault('maxDepth'),
    'maxBins': bestmodel.getOrDefault('maxBins'),
    'maxIter': bestmodel.getOrDefault('maxIter')
}
print(f"best param of gbt (eval=auc) is {best_gbt_paramDict}")

prediction = cvModel_gbt_auc.transform(small_test_DF)
auc_gbt = eval_auc.evaluate(prediction)
print(f"AUC of gbt using cvModel_ = {auc_gbt}")

# (shallow) Neural networks
#crossval_mpc2 = CrossValidator(estimator=pipeline_mpc,
#                             estimatorParamMaps=paramGrid_mpc,
#                             evaluator=eval_auc)
#cvModel_mpc_auc = crossval_mpc2.fit(small_train_DF)
#### select best model param
#bestmodel = cvModel_mpc_auc.bestModel.stages[-1]
#best_mpc_paramDict = {
#    'blockSize': bestmodel.getOrDefault('blockSize'),
#    'layers': bestmodel.getOrDefault('layers'),
#    'maxIter': bestmodel.getOrDefault('maxIter')
#}
#print(f"best param of mpc (eval=auc) is {best_mpc_paramDict}")
#prediction = cvModel_mpc_auc.transform(small_test_DF)
##test what prdiction looks like
#prediction.select('features', 'labels', 'rawPrediction').show(5)
#auc_mpc = eval_auc.evaluate(prediction)
#print(f"AUC of mpc using cvModel_ = {auc_mpc}")
