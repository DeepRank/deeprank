#!/bin/bash
#SBATCH -t 119:59:00
#SBATCH -p gpu
srun python train_001_smallNN_Kfold_balance_nofilt.py $SLURM_ARRAY_TASK_ID
