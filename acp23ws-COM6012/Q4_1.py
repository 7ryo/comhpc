##### ALS

# A. Recommender w/ ALS
# four-fold cross-validation ALS-based recommendation 
# data: ratings.csv
#       split into 4 folds
#       train:test = 75%:25%
#       repeat four times (manually)

# 3 versions of ALS
# setting1) use the settings in Lab8
#           change random seed = student number

# setting2)
# setting3)
# eg. changing rank, regParam, alpha
# improve the model

# Eval: 
# for each split - mean RMSE and mean MAE for 3 ALSs
# over four splits -
# [mean&standard deviation] of RMSE and MAE
# report => put all 36 numbers in a table

# plot => mean&std of RMSE and MAE for each of 3ver ALS

# ===================================================== #
# B. K-Means