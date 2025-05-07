#!/bin/bash
#SBATCH --job-name=training
#SBATCH --time=00:30:00     # limit
#SBATCH --nodes=10          # limit
#SBATCH --mem=10G           # limit
#SBATCH --output=./Output/Q3_output_training.txt 

module load Java
module load Anaconda3

source activate myspark

MODEL=$1
DATASIZE=$2

spark-submit ./trainModel.py --model "$MODEL" --datasize "$DATASIZE"