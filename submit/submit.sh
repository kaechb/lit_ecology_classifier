#!/bin/bash
#SBATCH --partition=allgpu,maxgpu
#SBATCH --constraint='GPUx4&V100'
#SBATCH --time=72:00:00                           # Maximum time requested
#SBATCH --nodes=1                          # Number of nodes
#SBATCH --chdir=/home/kaechben/slurm_thesis        # directory must already exist!
#SBATCH --job-name=hostname
#SBATCH --output=%j.out               # File to which STDOUT will be written
#SBATCH --error=%j.err                # File to which STDERR will be written
#SBATCH --mail-type=NONE

export OMP_NUM_THREADS=12 #$SLURM_CPUS_PER_TASK
cd /home/kaechben/Plankiformer
source /etc/profile.d/modules.sh
module purge
module load maxwell gcc/9.3
module load anaconda3/5.2
. conda-init
conda activate eawag
python -m lit_plankformer.main --use_wandb --max_epochs 20 --dataset zoo