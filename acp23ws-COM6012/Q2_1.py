# Install dependencies as needed:
# pip install kagglehub[pandas-datasets]
import kagglehub
from kagglehub import KaggleDatasetAdapter
# Set the path to the file you'd like to load
# diabetic_data.csv	18.3 MB
# IDS_mapping.csv     2.5 KB <- not accessible?
file_path = "diabetic_data.csv"

# Load the latest version
df = kagglehub.load_dataset(
KaggleDatasetAdapter.PANDAS,
"brandao/diabetes",
path=file_path,
# Provide any additional arguments like
# sql_query or pandas_kwargs. See the
# documenation for more information:
)
print("First 5 records:", df.head())

# =================== pySpark ======================== #
import json
from pyspark.sql import SparkSession
import pyspark.sql.functions as F
#from pyspark.pandas import DataFrame
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.tuning import CrossValidator, ParamGridBuilder
from pyspark.ml.regression import GeneralizedLinearRegression
from pyspark.ml.classification import LogisticRegression
from pyspark.ml.evaluation import RegressionEvaluator

spark = SparkSession.builder\
        .master("local[2]")\
        .appName("Question2")\
        .config("spark.local.dir", "/mnt/parscratch/users/acp23ws")\
        .getOrCreate()

sc = spark.sparkContext
sc.setLogLevel("ERROR")

# A. Preprocessing
# 1. one-hot encode medication features
features = ['metformin', 'repaglinide', 'nateglinide', 'chlorpropamide', 'glimepiride', 
            'acetohexamide', 'glipizide', 'glyburide', 'tolbutamide', 'pioglitazone', 
            'rosiglitazone', 'acarbose', 'miglitol', 'troglitazone', 'tolazamide', 
            'examide', 'citoglipton', 'insulin', 'glyburide-metformin', 'glipizide-metformin', 
            'glimepiride-pioglitazone', 'metformin-rosiglitazone', 'metformin-pioglitazone']

# convert df to spark df
spark_df = spark.createDataFrame(df)
#spark_df.show(5)

# 
new_spark_df = spark_df.select(*features, 'readmitted', 'time_in_hospital')
#new_spark_df.show(5)

# convert feature classes into numeric
for f in features:
	new_spark_df = new_spark_df.withColumn(f, F.when(F.col(f)=='No', 0).otherwise(1))

new_spark_df.show(5)

vecAssembler = VectorAssembler(inputCols=features, outputCol='onehot')
onehot_df = vecAssembler.transform(new_spark_df)
onehot_df = onehot_df.drop(*features)
onehot_df.show(5, False) # sparse display

#features_ohe = [f"{f}_ohe" for f in features]
#onehot_encoder = OneHotEncoder(inputCols=features, outputCols=features_ohe)
#onehot_model = onehot_encoder.fit(new_spark_df)
#onehot_df = onehot_model.transform(new_spark_df).drop(*features)
#onehot_df.printSchema()

#onehot_df.show(5)

#onehot_df.printSchema()

# 2. convert "readmitted" to binary
#    >30 and <30: 1
#             No: 0
onehot_df = onehot_df.withColumn('readmitted', F.when(F.col('readmitted').contains('NO'), 0.0).otherwise(1.0))
print(onehot_df.select('readmitted').groupBy('readmitted').count())

#onehot_df.printSchema()

# 3. select numeric feature from either 
# “time_in_hospital” or "num_lab_procedures”
# choose time_in_hospital

# =========================================== #
# B. split dataset into train set (8:2
#    seef = last five digits of your registration number
#         = 66896
#    ** use a stratified split on readmitted
trainData = onehot_df.sampleBy('readmitted', fractions={0: 0.8, 1: 0.8}, seed=66896)
trainData.show(5)
# ref https://stackoverflow.com/questions/47637760/stratified-sampling-with-pyspark
testData = onehot_df.subtract(trainData)


# C. train models
# cross validation -> OPT param
# regParam (out of [0.001,0.01, 0.1, 1, 10, 100]), 
# and elasticNetParam (out of [0, 0.2, 0.5, 0.8, 1])

# a) Poisson Regression ->
#    predict time_in_hospital or num_lab_procedures


print("model training")
glm_poisson = GeneralizedLinearRegression(featuresCol='onehot', labelCol='time_in_hospital', maxIter=50,\
                                          family='poisson', link='log')

paramGrid_glm = ParamGridBuilder()\
                .addGrid(glm_poisson.regParam, [0.001,0.01, 0.1, 1, 10, 100])\
                .build()
#    Eval: RMSE
evaluator_glm = RegressionEvaluator(labelCol="time_in_hospital", predictionCol="prediction", metricName="rmse")

crossval_glm = CrossValidator(estimator=glm_poisson,
			      estimatorParamMaps=paramGrid_glm,
			      evaluator=evaluator_glm)

cvModel_glm = crossval_glm.fit(trainData)

#### extract Metrics
avgMetrics_glm = cvModel_glm.avgMetrics
stdMetrics_glm = cvModel_glm.stdMetrics

import matplotlib.pyplot as plt
plt.errorbar([0.001,0.01, 0.1, 1, 10, 100], avgMetrics_glm, \
		yerr=stdMetrics_glm, fmt='-o')
plt.xscale('log')
plt.xlabel('regParam')
plt.ylabel('avg Accuracy')
plt.title('Errorbar of Poisson')
plt.savefig('glm_errorbar.png')
plt.close()

#### select the best model
best_glm = cvModel_glm.bestModel
best_glm_params = {
     'regParam': best_glm.getOrDefault('regParam')
}
print(f"Params of best glm poisson is {best_glm_params}")

#### train final model

best_glm_model = GeneralizedLinearRegression(featuresCol='onehot', labelCol='time_in_hospital', maxIter=50,\
                                          family='poisson', link='log',
                                          regParam=best_glm_params['regParam'])
final_glm_model = best_glm_model.fit(trainData)

prediction_glm = final_glm_model.transform(testData)
rmse_glm = evaluator_glm.evaluate(prediction_glm)

print(f"RMSE for final GLM poisson model = {rmse_glm}")

#model = glm_poisson.fit(trainData)
#predictions = model.transform(testData)

# ========================================================== #

# b) Logistic Regression -> 
#    model readmitted (binary classification) 
#    with elastic-net and L2 regularization
#    Eval: accuracy

from pyspark.ml.classification import LogisticRegression
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.ml.evaluation import BinaryClassificationEvaluator

# evaluator
acc_logit = MulticlassClassificationEvaluator(labelCol='readmitted', predictionCol='prediction', metricName='accuracy')
####auc_logit = BinaryClassificationEvaluator(labelCol='readmitted', rawPredictionCol='rawPrediction')

# L2: elasticNetParam=0
logit_l2 = LogisticRegression(featuresCol='onehot', labelCol='readmitted', \
				        maxIter=50, elasticNetParam=0)
paramGrid_l2 = ParamGridBuilder()\
                .addGrid(logit_l2.regParam, [0.001,0.01, 0.1, 1, 10, 100])\
		.build()
crossval_l2 = CrossValidator(estimator=logit_l2,
			     estimatorParamMaps=paramGrid_l2,
			     evaluator=acc_logit)
cvModel_l2 = crossval_l2.fit(trainData)
#####prediction = cvModel_l2.transform(testData)

avgMetrics_l2 = cvModel_l2.avgMetrics
stdMetrics_l2 = cvModel_l2.stdMetrics
plt.errorbar(x=[0.001,0.01, 0.1, 1, 10, 100], y=avgMetrics_l2,\
		yerr=stdMetrics_l2, fmt='-o')
plt.xscale('log')
plt.xlabel('regParam')
plt.ylabel('avg AUC')
plt.title('Errorbar of Logistic Regression w/ L2 reg')
plt.savefig('l2_errorbar.png')
plt.close()

#### best l2
#### select the best model
best_l2 = cvModel_l2.bestModel
best_l2_params = {
     'regParam': best_l2.getOrDefault('regParam')
}
print(f"Params of best LR l2 is {best_l2_params}")

best_l2_model = LogisticRegression(featuresCol='onehot', labelCol='readmitted', \
				        maxIter=50, elasticNetParam=0,\
                                        regParam=best_l2_params['regParam'])
final_l2_model = best_l2_model.fit(trainData)

prediction_l2 = final_l2_model.transform(testData)
acc_l2 = acc_logit.evaluate(prediction_l2)

print(f"Accuracy for final LR l2 model = {acc_l2}")


# with elasticNetParam
logit_elastic = LogisticRegression(featuresCol='onehot', labelCol='readmitted', \
				        maxIter=50)
elas_netparam_list = [0, 0.2, 0.5, 0.8, 1]

paramGrid_elastic = ParamGridBuilder()\
			.addGrid(logit_elastic.elasticNetParam, elas_netparam_list)\
			.addGrid(logit_elastic.regParam, [0.001,0.01, 0.1, 1, 10, 100])\
			.build() ## when elasticNetParam==0 -> L2
crossval_elastic = CrossValidator(estimator=logit_elastic,
                                  estimatorParamMaps=paramGrid_elastic,
				  evaluator=acc_logit)

cvModel_elastic = crossval_elastic.fit(trainData)
avgMetrics_elastic = cvModel_elastic.avgMetrics
stdMetrics_elastic = cvModel_elastic.stdMetrics

from collections import defaultdict
results = defaultdict(list)
reg_params = [0.001, 0.01, 0.1, 1, 10, 100]
elas_netparam_list = [0, 0.2, 0.5, 0.8, 1]

### remember to modify this
for i, param_map in enumerate(paramGrid_elastic):
    reg = param_map[logit_elastic.regParam]
    enet = param_map[logit_elastic.elasticNetParam]
    mean = avgMetrics_elastic[i]
    std = stdMetrics_elastic[i]
    results[enet].append((reg, mean, std))

for enet in elas_netparam_list:
    data = sorted(results[enet])  # Sort by regParam
    regs, means, stds = zip(*data)
    plt.errorbar(regs, means, yerr=stds, fmt='-o', capsize=3, label=f'elasticNetParam={enet}')

#for i in range(len(elas_netparam_list)): #[0, 0.2, 0.5, 0.8, 1]
#	plt.errorbar(x=[0.001,0.01, 0.1, 1, 10, 100], y=avgMetrics_elastic[i],
#			yerr=stdMetrics_glm[i], fmt='-o', 
#			label=f"elastic={elas_netparam_list[i]}")

plt.xscale('log')
plt.xlabel('regParam')
plt.ylabel('avg AUC')
plt.legend()
plt.title('Logistic Regresion w/ elasticnet')
plt.savefig('elas_errorbar.png')
plt.close()

#### best elas
#### select the best elas model
best_elastic = cvModel_elastic.bestModel
best_elastic_params = {
     'regParam': best_elastic.getOrDefault('regParam'),
     'elastic': best_elastic.getOrDefault('elasticNetParam')
}
print(f"Params of best LR elastic is {best_elastic_params}")

best_elastic_model = LogisticRegression(featuresCol='onehot', labelCol='readmitted', \
				        maxIter=50,\
                                        elasticNetParam=best_elastic_params['elastic'],\
                                        regParam=best_elastic_params['regParam'])
final_elastic_model = best_elastic_model.fit(trainData)

prediction_elastic = final_elastic_model.transform(testData)
acc_elastic = acc_logit.evaluate(prediction_elastic)

print(f"Accuracy for final LR elastic model = {acc_elastic}")





# model_l2 = logit_l2.fit(trainData)
# pred_l2 = model_l2.transform(testData)

# #check prediction output
# pred_l2.show(10)

# model_elastic = logit_elastic.fit(trainData)
# pred_elastic = model_elastic.transform(testData)
# pred_elastic.show(10)


# acc_l2 = evaluator_logit.evaluate(pred_l2)
# print(f"Accuracy of l2 is {acc_l2}\n")

# ###manul check
# #correct = pred_l2.filter("readmitted = prediction").count()
# #total = pred_l2.count()
# #print("Manual Accuracy: ", correct / total)


# acc_elastic = evaluator_logit.evaluate(pred_elastic)
# print(f"Accuracy of w/ elastic is {acc_elastic}\n")
