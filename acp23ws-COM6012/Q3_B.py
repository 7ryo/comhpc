from pyspark.sql import SparkSession
import pyspark.sql.functions as F
from pyspark.sql.types import StringType

spark = SparkSession.builder\
        .master("local[2]")\
        .appName("Question3")\
        .getOrCreate()

sc = spark.sparkContext
sc.setLogLevel("ERROR")

# Runtime log will be saved to an explicit .csv file for later plotting.
# The columns of each row will be
#   model:      rf, gbt, mpc
#   dataSize:   x portion of the whole data
#   runtime:    init = 30min
#               if the training is within 30min,
#               the value will be replaced with the actual runtime
#   accuracy:   comptued using MulticlassClassificationEvaluator
#   auc:        computed using BinaryClassificationEvaluator


# 1. Initialize the .csv
model_list = ['RF', 'GBT', 'MPC']
####size_list = [0.05, 0.1, 0.3, 0.5, 0.7, 0.8, 0.9, 1]
size_list = [0.1, 0.5, 0.7, 0.8, 0.9, 1.0]

# data = [(model, float(size), 1800, 0.0, 0.0) for model in model_list for size in size_list]
# init_DF = spark.createDataFrame(data, schema=['model', 'datasize', 'runtime', 'accuracy', 'auc'])

# init_DF.toPandas().to_csv('./log_runtime.csv', header=True)

# 2. Pass model and datasize to .sh to submit jobs
import subprocess
for model in model_list:
	for size in size_list:
		subprocess.run(["sbatch","submitTrain.sh", model, str(size)])
