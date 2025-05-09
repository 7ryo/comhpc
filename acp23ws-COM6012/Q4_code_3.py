##### ALS
from pyspark.sql import SparkSession, Row
spark = SparkSession.builder\
        .master("local[2]")\
        .appName("Question2")\
        .config("spark.local.dir", "/mnt/parscratch/users/acp23ws")\
        .getOrCreate()

sc = spark.sparkContext
sc.setLogLevel("ERROR")
# A. Recommender w/ ALS
# four-fold cross-validation ALS-based recommendation 
# ==================================================== #
# data: ratings.csv
#       split into 4 folds
#       train:test = 75%:25%
#       repeat four times (manually)
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml.recommendation import ALS
###from pyspark.sql import Row
import pyspark.sql.functions as F

# load data from csv
raw_rating_DF = spark.read.csv('./Data/ml-25m/ratings.csv', header=True).cache()
raw_rating_DF.show(5)
print(f"there are {raw_rating_DF.count()} rows of data in ratings.csv\n")
raw_rating_DF = raw_rating_DF.withColumn('userId', F.col('userId').cast("long"))
raw_rating_DF = raw_rating_DF.withColumn('movieId', F.col('movieId').cast("long"))
raw_rating_DF = raw_rating_DF.withColumn('rating', F.col('rating').cast("double"))

# check data type
raw_rating_DF.printSchema()

rand_seed = 1840977

# reduce the size
# raw_rating_DF = raw_rating_DF.sample(fraction=0.5, seed=rand_seed)

# pre processing
splits = raw_rating_DF.randomSplit([0.25, 0.25, 0.25, 0.25], seed=rand_seed)

# setting1) use the settings in Lab8
#           change random seed = student number
als_1 = ALS(userCol="userId", itemCol="movieId", \
		seed=rand_seed, coldStartStrategy="drop")

itemFactors_dict = {}

for i in range(4):
    print(f"i={i}")
    test = splits[i].cache()
    training = raw_rating_DF.subtract(test).cache()

    model_1 = als_1.fit(training)
    itemFactors_dict[i] = model_1.itemFactors

    test.unpersist()
    training.unpersist()

#### itemFactors is a df
itemFactors_dict[0].show(10)

# ===================================================== #
# B. K-Means

from pyspark.ml.clustering import KMeans
from pyspark.ml.clustering import KMeansModel
import matplotlib.pyplot as plt

movie_tag_DF = spark.read.csv('./Data/ml-25m/genome-scores.csv', header=True)
tag_DF = spark.read.csv('./Data/ml-25m/genome-tags.csv', header=True)

kmeans = KMeans(k=19, seed=rand_seed)

# try with the model of first split itemFactors_dict[0]
tags_dict = {}
for i in range(4):
    print(f"split{i}")
    tags_per_split_dict = {}
    model_kmeans = kmeans.fit(itemFactors_dict[i])
    # save model
    # model_kmeans.save(f'./kmeans_model_{i}')

    transformed = model_kmeans.transform(itemFactors_dict[0])

    top_clusters = transformed.groupBy("prediction").count()\
                                .orderBy(F.desc("count")).limit(3)
    top_clusters_id_list = [row['prediction'] for row in top_clusters.select('prediction').collect()]
    print(f"top 3 clusters: {top_clusters_id_list}")

    # within each cluster
    for m in top_clusters_id_list:
        print(f"cluster {i}:")
        movie_tag_in_cluster = transformed.filter(F.col('prediction')==m).select('id')\
                                        .join(movie_tag_DF, movie_tag_DF.movieId == transformed.id)
        movie_tag_in_cluster.show(5)

        # top tag id -> names
        top_tags = movie_tag_in_cluster.groupBy(F.col('tagId'))\
                                        .agg(F.sum('relevance').alias('sum_score'))\
                                        .orderBy('sum_score').limit(3)\
                                        .join(tag_DF, on="tagId")
        top_tags.show()
        top_tags_list = [row['tag'] for row in top_tags.select('tag').collect()]
        tags_per_split_dict[m] = top_tags_list
    tags_dict[i] = tags_per_split_dict
