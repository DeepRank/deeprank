To train using a slurm job scheduler with 10-fold cross validation one needs to use the command 

sbatch --array=1-10 kfold_train_001_3_2806-prealigned_rotated.slurm

Alternatively, you can start the script train_001_3_2806_prealigned_rotated.py 10 times with 

python train_001_3_2806_prealigned_rotated.py n

where n runs through 1,2,...10.

The training script assumes one GPU card.

The script model_280619.py needs to be also in the same folder as train_001_3_2806_prealigned_rotated.py

The output is uploaded at http://doi.org/10.5281/zenodo.3953965