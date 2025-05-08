from pyspark.sql import SparkSession
from pyspark.sql.functions import col
spark = SparkSession.builder\
        .master("local[2]")\
        .appName("Question1")\
        .config("spark.local.dir", "/mnt/parscratch/users/acp23ws")\
        .getOrCreate()

sc = spark.sparkContext
sc.setLogLevel("WARN")

logFile = spark.read.text("./Data/NASA_access_log_Jul95.gz").cache()

# United States (host ending in “.edu”)
# United Kingdom (host ending in “.ac.uk”)
# Australia (host ending in “.edu.au”)

hostUS = logFile.filter(logFile.value.contains(".edu ")) #end
hostUK = logFile.filter(logFile.value.contains(".ac.uk"))
hostAU = logFile.filter(logFile.value.contains(".edu.au"))

print("Hello from spark!")
print("\n\n")
print("Hosts from US: %i.\n" % hostUS.count())
hostUS.show(20, False)
print("Hosts from UK: %i.\n" % hostUK.count())
hostUK.show(20, False)
print("Hosts from Australia: %i.\n" % hostAU.count())
hostAU.show(20, False)


spark.stop()
