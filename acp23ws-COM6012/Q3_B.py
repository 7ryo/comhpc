# from pyspark.sql import SparkSession
# spark = SparkSession.builder\
#         .master("local[2]")\
#         .appName("Question3")\
#         .getOrCreate()

# sc = spark.sparkContext
# sc.setLogLevel("ERROR")

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
size_list = [0.05, 0.1, 0.3, 0.5, 0.7, 0.8, 0.9, 1]

# data = [(model, float(size), 1800, 0.0, 0.0) for model in model_list for size in size_list]
# init_DF = spark.createDataFrame(data, schema=['model', 'datasize', 'runtime', 'accuracy', 'auc'])

# init_DF.toPandas().to_csv('./log_runtime.csv', header=True)

# 2. Pass model and datasize to .sh to submit jobs
# import subprocess
# for model in model_list:
# 	for size in size_list:
# 		subprocess.run(["sbatch","submitTrain.sh", model, str(size)])

# 3. Plot the performance
import matplotlib.pyplot as plt
import pandas as pd
#log_DF = spark.read.csv('./log_runtime.csv', header=True)
log_DF = pd.read_csv('./log_runtime.csv', index_col=0)

plt.figure(figsize=(13,5))
plt.subplot(1, 3, 1)
for m in model_list:
        sub_DF = log_DF[log_DF['model']==m]
        val_list = sub_DF['accuracy']
        plt.plot(size_list, val_list, label=m)
plt.xlabel('Datasize')
plt.ylabel('Accuracy')
plt.title('Accuracy')
plt.legend()

plt.subplot(1, 3, 2)
for m in model_list:
        sub_DF = log_DF[log_DF['model']==m]
        val_list = sub_DF['auc']
        plt.plot(size_list, val_list, label=m)
        #plt.xticks(size_list)
plt.xlabel('Datasize')
plt.ylabel('Area Under the Curve')
plt.title('Area Under the Curve')
plt.legend()

plt.subplot(1, 3, 3)
for m in model_list:
        sub_DF = log_DF[log_DF['model']==m]
        val_list = sub_DF['runtime'] / 60
        plt.plot(size_list, val_list, label=m)
        #plt.xticks(size_list)
plt.xlabel('Datasize')
plt.ylabel('Runtime: minutes')
plt.title('Runtime')
plt.legend()
plt.tight_layout()
plt.savefig('./log_auc.png')