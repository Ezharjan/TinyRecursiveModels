#!/bin/bash
#SBATCH --job-name=trm
#SBATCH --partition=large-gpu   # Using large-gpu partition with many idle GPUs
#SBATCH --nodes=1               # 1 node
#SBATCH --ntasks-per-node=4     # Processes per node (one per GPU)
#SBATCH --gpus-per-node=4       # GPUs per node
#SBATCH --time=7-00:00:00       # Max time limit
#SBATCH --output=trm_%j.out
#SBATCH --error=trm_%j.err
#SBATCH --mem=0                 # Request all available memory per node

# Load cluster modules
# module load cuda/12.6  # Not available, using environment CUDA
# module load python/3.10  # Not available, using environment Python

# Initialize conda
source ~/anaconda3/etc/profile.d/conda.sh

# Activate your Conda environment
conda activate trm

# Change to your code directory
cd /SEAS/home/g28630518/proj/TinyRecursiveModels

# Run ARC-AGI-1 training
run_name="pretrain_att_arc1concept_4"
torchrun \
  --nnodes=1 \
  --nproc_per_node=4 \
  --rdzv_id=$SLURM_JOB_ID \
  --rdzv_backend=c10d \
  --rdzv_endpoint=localhost:29500 \
  pretrain.py \
  arch=trm \
  data_paths="[data/arc1concept-aug-1000]" \
  arch.L_layers=2 \
  arch.H_cycles=3 arch.L_cycles=4 \
  +run_name=${run_name} ema=True