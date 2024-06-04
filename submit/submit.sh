#!/bin/bash
#SBATCH --account="ma"
#SBATCH --constraint='gpu'
#SBATCH --nodes=1
#SBATCH --ntasks-per-core=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=12
#SBATCH --partition=normal
#SBATCH --constraint=gpu
#SBATCH --hint=nomultithread
#SBATCH --output=slurm/slurm_%j.out
#SBATCH --error=slurm/slurm_%j.err
#SBATCH --time=6:00:00
export OMP_NUM_THREADS=12 #$SLURM_CPUS_PER_TASK
cd ${SCRATCH}/lit_ecology_classifier
module purge
module load daint-gpu cray-python
source lit_ecology/bin/activate
python -m lit_ecology_classifier.main --max_epochs 20 --dataset phyto 