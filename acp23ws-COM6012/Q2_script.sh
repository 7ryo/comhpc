#!/bin/bash
#SBATCH --job-name=JOB_NAME  
#SBATCH --time=00:30:00  
#SBATCH --nodes=4  
#SBATCH --mem=16G  
#SBATCH --output=./Output/Q2_output.txt  
module load Java
module load Anaconda3

source activate myspark

spark-submit ./Q2_code.py
