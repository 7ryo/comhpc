#!/bin/bash
#SBATCH --job-name=JOB_NAME  # Replace JOB_NAME with a name you like
#SBATCH --time=00:30:00  # Change this to a longer timore time
#SBATCH --nodes=2  # Specify a number of nodes
#SBATCH --mem=8G  # Request 4 gigabytes of real memory (mem)
#SBATCH --output=./Output/Q2_output.txt  # This is where your output and errors are logged

module load Java
module load Anaconda3

source activate myspark

spark-submit ./Q2_1.py
