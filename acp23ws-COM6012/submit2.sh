#!/bin/bash
#SBATCH --job-name=JOB_NAME  # Replace JOB_NAME with a name you like
#SBATCH --time=02:50:00  # Change this to a longer timore time
##SBATCH --account=rse-com6012
##SBATCH --reservation=rse-com6012-9  # Replace $LAB_ID with your lab session number
#SBATCH --nodes=4  # Specify a number of nodes
#SBATCH --mem=100G  # Request 4 gigabytes of real memory (mem)
#SBATCH --output=./Output/Q4_larger.txt  # This is where your output and errors are logged

module load Java
module load Anaconda3

source activate myspark

spark-submit ./Q4_code_3.py
