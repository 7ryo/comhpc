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

spark_df = spark.createDataFrame(df)
spark_df.show(5)
onehot_df = spark_df.select('encounter_id')
for f in features[:3]:
    onehot_df = spark_df.withColumn(f, F.when(F.col(f)=="No", 0)\
                                    .otherwise(1)).cache()
onehot_df.head(10)

# 2. convert "readmitted" to binary
#    >30 and <30: 1
#             No: 0

# 3. select numeric feature from either 
# “time_in_hospital” or "num_lab_procedures”


# =========================================== #
# B. split dataset into train set (8:2
#    seef = last five digits of your registration number
#         = 66896
#    ** use a stratified split on readmitted

# C. train models
# a) Poisson Regression ->
#    predict time_in_hospital or num_lab_procedures

# b) Logistic Regression -> 
# model readmitted (binary classification) 
# with elastic-net and L2 regularization