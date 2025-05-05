from pyspark.sql import SparkSession
import pyspark.sql.functions as F
from pyspark.sql.window import Window
import matplotlib.pyplot as plt
import seaborn as sns

spark = SparkSession.builder\
        .master("local[2]")\
        .appName("Question1")\
        .config("spark.local.dir", "/mnt/parscratch/users/acp23ws")\
        .getOrCreate()

sc = spark.sparkContext
sc.setLogLevel("WARN")

logFile = spark.read.text("./Data/NASA_access_log_Jul95.gz").cache()

# convert logFile into DataFrame based on the five columns

# example log:
# in24.inetnebr.com - - [01/Aug/1995:00:00:01 -0400] "GET /shuttle/missions/sts-68/news/sts-68-mcc-05.txt HTTP/1.0" 200 1839
# in the order of host, timestamp, request, http reply code, bytes in the reply

# use regex to extract each part
# 

pattern = r'^(.*?) - - \[(.*?)\] "(.*?)" (\d+) (\d+)'

df = logFile.withColumn('host', F.regexp_extract('value', pattern, 1))\
            .withColumn('timestamp', F.regexp_extract('value', pattern, 2))\
            .withColumn('request', F.regexp_extract('value', pattern, 3))\
            .withColumn('HTTP reply code', F.regexp_extract('value', pattern, 4))\
            .withColumn('bytes in the reply', F.regexp_extract('value', pattern, 5))\
            .withColumn('country', F.when(F.col('host').endswith('.ac.uk'), "UK")\
                                    .when(F.col('host').endswith('.edu'), "US")\
                                    .when(F.col('host').contains('.edu.au'), "Australia")\
                                    .otherwise("other"))\
            .drop('value').cache()

df.show(10, False)

# count total number of unique instituitions
num_host = df.select('host').distinct().count()
print("total number of distinct hosts is %i.\n" % num_host)


# top 9 most frequent visitors (9 per country)
#hostcount_uk = df.select('host').groupBy('')
counted_host = df.select('country', 'host').groupBy('country', 'host').count().sort('count', ascending=False).cache()

print("Count of all hosts")
counted_host.show(10, False)

print("Most frequent 9 hosts per country")
window_spec = Window.partitionBy('country').orderBy(F.col('count').desc())
ranked_host = counted_host.withColumn("rank", F.rank().over(window_spec)).cache()
ranked_host.show(10, False)

top9_host = ranked_host.filter(F.col('rank') <= 9).orderBy('country', 'rank').cache()
top9_host.show(27, False)



# find out the ranking of Univerisity of Sheffield (UK)
# .shef.ac.uk
print("University of Sheffield:")
ranked_host.filter(F.col('host').contains('.shef.ac.uk')).show(30)

# visualize: show top 9 and "other"
# sum hosts ranked after 9
## agg(sum(col('count')).alias('count')).withColumn('host', lit('other'))
rear = ranked_host.filter(F.col('rank') > 9)\
                    .groupBy('country')\
                    .agg(F.sum(F.col('count')).alias('count'))\
                    .withColumn('host', F.lit('other institutions'))\
			.withColumn('rank', F.lit(0)).cache()

# combine top 9 and "others"
unioned = top9_host.unionByName(rear, allowMissingColumns=True).cache()
print("unioned")
unioned.show(41)

for country in ['UK', 'US', 'Australia']:
    unioned_c = unioned.select('host', 'count').filter(F.col('country')==country).cache()
    unioned_c.show(20)
    pd_df = unioned_c.toPandas()
    
    plt.figure(figsize=(14, 8))
    plt.pie(pd_df['count'], explode=[0.1]*len(pd_df))
    plt.title(f"Pie chart of {country}")
    legend_txt = [f'{host}: {count}' for host, count in zip(pd_df['host'], pd_df['count'])]
    plt.legend(legend_txt, loc='upper left', bbox_to_anchor=(1.5, 0.5))
    plt.tight_layout()
    plt.savefig(f"{country}.png")

    plt.close()    

# select top institution of each country
top_hosts = ranked_host.filter((F.col('rank')==1) & (F.col('country')!='other'))\
            .select('host').distinct()
top_hosts.show()

top_df = df.join(top_hosts, 'host').select('country', 'host', 'timestamp').cache()
top_df.show()




# convert str to timestamp
# [01/Aug/1995:00:00:01 -0400]
top_df = top_df.withColumn('to_timestamp', F.to_timestamp(F.col('timestamp'), 'dd/MMM/yyyy:HH:mm:ss Z')).cache()
top_df.show()
top_df = top_df.withColumn('day of month', F.dayofmonth('to_timestamp'))\
        .withColumn('hour', F.hour('to_timestamp'))\
            .drop('timestamp').cache()
top_df.show()

heatmap_df = top_df.groupBy('country', 'day of month', 'hour').count().cache()
heatmap_df.show(10)
#sns.heatmap()
for country in ['UK', 'US', 'Australia']:
    
    country_df = heatmap_df.filter(F.col('country')==country).toPandas()
    country_pivot = country_df.pivot(index='day of month', columns='hour', values='count')

    
    #plt.figure(figsize=(14, 8))
    sns.heatmap(country_pivot, annot=True)
    #plt.title(f"Pie chart of {country}")
    #legend_txt = [f'{host}: {count}' for host, count in zip(pd_df['host'], pd_df['count'])]
    #plt.legend(legend_txt, loc='upper left', bbox_to_anchor=(1.5, 0.5))
    plt.tight_layout()
    plt.savefig(f"{country}_heatmap.png")

    plt.close()    



spark.stop()

