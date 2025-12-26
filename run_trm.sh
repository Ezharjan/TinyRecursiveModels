#!/bin/bash
#SBATCH --job-name=TRM_ARC1
#SBATCH --partition=large-gpu   # CHANGED: Switched to the available high-performance partition
#SBATCH --nodes=2               # Keeps 2 nodes (there are 10 idle in large-gpu)
#SBATCH --ntasks-per-node=4     # Processes per node (one per GPU)
#SBATCH --gpus-per-node=4       # GPUs per node (Fits standard large-gpu nodes)
#SBATCH --time=7-00:00:00       # Max time limit matches large-gpu limit
#SBATCH --output=trm_%j.out
#SBATCH --error=trm_%j.err
#SBATCH --mem=0                 # Request all available memory per node

# #SBATCH --nodelist=hth[005-006] # DISABLED: Let SLURM pick from the 10 idle nodes (gpu001-gpu010)

# Load cluster modules if required (e.g., for CUDA/Python)
# module load cuda/12.6
# module load python/3.10

# Activate your Conda environment
conda activate trm

# Set up distributed training variables
export MASTER_PORT=29500
nodes=$(scontrol show hostnames "$SLURM_JOB_NODELIST")
nodes_array=(${nodes})
export MASTER_ADDR=${nodes_array[0]}

# Change to your code directory
cd /path/to/your/code

# Run the training command
run_name="pretrain_att_arc1concept_4"
torchrun \
  --nnodes=$SLURM_JOB_NUM_NODES \
  --nproc_per_node=4 \
  --rdzv_id=$SLURM_JOB_ID \
  --rdzv_backend=c10d \
  --rdzv_endpoint=$MASTER_ADDR:$MASTER_PORT \
  pretrain.py \
  arch=trm \
  data_paths="[data/arc1concept-aug-1000]" \
  arch.L_layers=2 \
  arch.H_cycles=3 arch.L_cycles=4 \
  +run_name=${run_name} ema=True