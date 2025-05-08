#!/bin/bash
#SBATCH --job-name=JOB_NAME  # Replace JOB_NAME with a name you like
#SBATCH --time=00:30:00  # Change this to a longer timore time
##SBATCH --account=rse-com6012
##SBATCH --reservation=rse-com6012-9  # Replace $LAB_ID with your lab session number
#SBATCH --nodes=10  # Specify a number of nodes
#SBATCH --mem=10G  # Request 4 gigabytes of real memory (mem)
#SBATCH --output=./Output/Q3_output.txt  #  where your output

module load Java
module load Anaconda3

source activate myspark

spark-submit ./Q3_code_1.py
