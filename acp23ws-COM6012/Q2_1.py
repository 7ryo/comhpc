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

from pyspark.sql import SparkSession
import pyspark.sql.functions as F
#from pyspark.pandas import DataFrame
from pyspark.ml.feature import VectorAssembler

spark = SparkSession.builder\
        .master("local[2]")\
        .appName("Question2")\
        .config("spark.local.dir", "/mnt/parscratch/users/acp23ws")\
        .getOrCreate()

sc = spark.sparkContext
sc.setLogLevel("WARN")

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
new_spark_df = vecAssembler.transform(new_spark_df)
new_spark_df.show(5, False) # sparse display

#features_ohe = [f"{f}_ohe" for f in features]
#onehot_encoder = OneHotEncoder(inputCols=features, outputCols=features_ohe)
#onehot_model = onehot_encoder.fit(new_spark_df)
#onehot_df = onehot_model.transform(new_spark_df).drop(*features)
#onehot_df.printSchema()

#onehot_df.show(5)

new_spark_df.printSchema()

# 2. convert "readmitted" to binary
#    >30 and <30: 1
#             No: 0
new_spark_df = new_spark_df.withColumn('readmitted', F.when(F.col('readmitted')=="NO", 0).otherwise(1))
new_spark_df.select('readmitted').show(5)


# 3. select numeric feature from either 
# “time_in_hospital” or "num_lab_procedures”
# choose time_in_hospital

# =========================================== #
# B. split dataset into train set (8:2
#    seef = last five digits of your registration number
#         = 66896
#    ** use a stratified split on readmitted
trainData = new_spark_df.sampleBy('readmitted', fractions={0: 0.8, 1: 0.8}, seed=66896)
trainData.show(5)
# ref https://stackoverflow.com/questions/47637760/stratified-sampling-with-pyspark
testData = new_spark_df.subtract(trainData)


# C. train models
# cross validation -> OPT param

# a) Poisson Regression ->
#    predict time_in_hospital or num_lab_procedures

from pyspark.ml.regression import GeneralizedLinearRegression
from pyspark.ml.evaluation import RegressionEvaluator

print("model training")
glm_poisson = GeneralizedLinearRegression(featuresCol='onehot', labelCol='time_in_hospital', maxIter=50, regParam=0.01,\
                                          family='poisson', link='log')
model = glm_poisson.fit(trainData)
predictions = model.transform(testData)

#    Eval: RMSE
evaluator = RegressionEvaluator(labelCol="time_in_hospital", predictionCol="prediction", metricName="rmse")
rmse = evaluator.evaluate(predictions)
print("RMSE = %g " % rmse)

# b) Logistic Regression -> 
#    model readmitted (binary classification) 
#    with elastic-net and L2 regularization
#    Eval: accuracy

