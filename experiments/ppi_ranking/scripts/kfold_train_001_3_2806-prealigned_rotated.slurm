#!/bin/bash
#SBATCH -t 29:59:00
#SBATCH -p gpu
srun python train_001_3_2806_prealigned_rotated.py $SLURM_ARRAY_TASK_ID
