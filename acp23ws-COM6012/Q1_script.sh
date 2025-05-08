#!/bin/bash
#SBATCH --job-name=JOB_NAME  
#SBATCH --time=00:30:00  # Change this to a longer timore time
#SBATCH --nodes=2  
#SBATCH --mem=8G  
#SBATCH --output=./Output/Q1_output.txt  

module load Java
module load Anaconda3

source activate myspark

spark-submit ./Q1_code.py
