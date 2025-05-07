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
prediction_glm = cvModel_glm.transform(testData)
rmse_glm = evaluator_glm.evaluate(prediction_glm)

print(f"RMSE for best lm model = {rmse_glm}")

#paramDict = {param[0].name: param[1] for param in cvModel_glm.bestModel.stages[-1].extractParamMap().items()}
#print(json.dumps(paramDict, indent=4))

#### extract Metrics
avgMetrics_glm = cvModel_glm.avgMetrics
stdMetrics_glm = cvModel_glm.stdMetrics

import matplotlib.pyplot as plt
plt.errorbar([0.001,0.01, 0.1, 1, 10, 100], avgMetrics_glm, \
		yerr=stdMetrics_glm, fmt='-o')
plt.xscale('log')
plt.title('Errorbar of Poisson')
plt.savefig('glm_errorbar.png')
plt.close()


#model = glm_poisson.fit(trainData)
#predictions = model.transform(testData)


#rmse = evaluator_poisson.evaluate(predictions)
#print("RMSE = %g \n" % rmse)

# b) Logistic Regression -> 
#    model readmitted (binary classification) 
#    with elastic-net and L2 regularization
#    Eval: accuracy

from pyspark.ml.classification import LogisticRegression
from pyspark.ml.evaluation import BinaryClassificationEvaluator
#from pyspark.ml.evaluation import MulticlassClassificationEvaluator

# evaluator
#evaluator_logit = MulticlassClassificationEvaluator(labelCol='readmitted', predictionCol='prediction', metricName='accuracy')
auc_logit = BinaryClassificationEvaluator(labelCol='readmitted', rawPredictionCol='rawPrediction')

# L2: elasticNetParam=0
logit_l2 = LogisticRegression(featuresCol='onehot', labelCol='readmitted', \
				        maxIter=50, regParam=0.1, elasticNetParam=0)
paramGrid_l2 = ParamGridBuilder()\
                .addGrid(logit_l2.regParam, [0.001,0.01, 0.1, 1, 10, 100])\
		.build()
crossval_l2 = CrossValidator(estimator=logit_l2,
			     estimatorParamMaps=paramGrid_l2,
			     evaluator=auc_logit)
cvModel_l2 = crossval_l2.fit(trainData)
#####prediction = cvModel_l2.transform(testData)

avgMetrics_l2 = cvModel_l2.avgMetrics
stdMetrics_l2 = cvModel_l2.stdMetrics
plt.errorbar(x=[0.001,0.01, 0.1, 1, 10, 100], y=avgMetrics_l2,\
		yerr=stdMetrics_l2, fmt='-o')
plt.xscale('log')
plt.title('Errorbar of Logistic Regression w/ L2 reg')
plt.savefig('l2_errorbar.png')
plt.close()



# with elasticNetParam
logit_elastic = LogisticRegression(featuresCol='onehot', labelCol='readmitted', \
				        maxIter=50, regParam=0.1, elasticNetParam=0.5)
paramGrid_elastic = ParamGridBuilder()\
                        .addGrid(logit_elastic.regParam, [0.001,0.01, 0.1, 1, 10, 100])\
			.addGrid(logit_elastic.elasticNetParam, [0, 0.2, 0.5, 0.8, 1])\
			.build() ## when elasticNetParam==0 -> L2
crossval_elastic = CrossValidator(estimator=logit_elastic,
                                  estimatorParamMaps=paramGrid_elastic,
				  evaluator=auc_logit)

cvModel_elastic = crossval_elastic.fit(trainData)
avgMetrics_elastic = cvModel_elastic.avgMetrics
stdMetrics_elastic = cvModel_elastic.stdMetrics
for i in len(logit_elastic.elasticNetParam): #[0, 0.2, 0.5, 0.8, 1]
	plt.errorbar(x=[0.001,0.01, 0.1, 1, 10, 100], y=avgMetrics_elastic[i],
			yerr=stdMetrics_glm[i], fmt='-o', 
			label=f"elastic={logit_elastic.elasticNetParam[i]}")
plt.xscale('log')
plt.legend()
plt.title('Logistic Regresion w/ elasticnet')
plt.savefig('elas_errorbar.png')
plt.close()


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
