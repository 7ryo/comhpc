#!/bin/bash
#SBATCH --job-name=JOB_NAME  
#SBATCH --time=02:50:00  
#SBATCH --nodes=4  
#SBATCH --mem=120G  
#SBATCH --output=./Output/Q4_output_tags.txt 

module load Java
module load Anaconda3

source activate myspark

spark-submit ./Q4_code_2.py
