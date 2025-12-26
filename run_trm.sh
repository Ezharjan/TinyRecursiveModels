#!/bin/bash
#SBATCH --job-name=TRM_ARC1
#SBATCH --partition=large-gpu   # Using large-gpu partition with many idle GPUs
#SBATCH --nodes=1               # Changed to 1 node to match README assumptions
#SBATCH --ntasks-per-node=4     # Processes per node (one per GPU)
#SBATCH --gpus-per-node=4       # GPUs per node
#SBATCH --time=7-00:00:00       # Max time limit
#SBATCH --output=trm_%j.out
#SBATCH --error=trm_%j.err
#SBATCH --mem=0                 # Request all available memory per node

# #SBATCH --nodelist=gpu[001-004] # Let SLURM pick from idle nodes

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
cd /SEAS/home/g28630518/proj/TinyRecursiveModels

# Build datasets from scratch
python -m dataset.build_arc_dataset \
  --input-file-prefix kaggle/combined/arc-agi \
  --output-dir data/arc1concept-aug-1000 \
  --subsets training evaluation concept \
  --test-set-name evaluation

python -m dataset.build_arc_dataset \
  --input-file-prefix kaggle/combined/arc-agi \
  --output-dir data/arc2concept-aug-1000 \
  --subsets training2 evaluation2 concept \
  --test-set-name evaluation2

python dataset/build_sudoku_dataset.py --output-dir data/sudoku-extreme-1k-aug-1000 --subsample-size 1000 --num-aug 1000

python dataset/build_maze_dataset.py

# Run ARC-AGI-1 training
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

# Run ARC-AGI-2 training
run_name="pretrain_att_arc2concept_4"
torchrun \
  --nnodes=$SLURM_JOB_NUM_NODES \
  --nproc_per_node=4 \
  --rdzv_id=$SLURM_JOB_ID \
  --rdzv_backend=c10d \
  --rdzv_endpoint=$MASTER_ADDR:$MASTER_PORT \
  pretrain.py \
  arch=trm \
  data_paths="[data/arc2concept-aug-1000]" \
  arch.L_layers=2 \
  arch.H_cycles=3 arch.L_cycles=4 \
  +run_name=${run_name} ema=True

# Run Sudoku-Extreme training (using 1 GPU, but allocated 4, still works)
run_name="pretrain_att_sudoku"
python pretrain.py \
  arch=trm \
  data_paths="[data/sudoku-extreme-1k-aug-1000]" \
  evaluators="[]" \
  epochs=50000 eval_interval=5000 \
  lr=1e-4 puzzle_emb_lr=1e-4 weight_decay=1.0 puzzle_emb_weight_decay=1.0 \
  arch.L_layers=2 \
  arch.H_cycles=3 arch.L_cycles=6 \
  +run_name=${run_name} ema=True

# Run Maze-Hard training (using 4 GPUs)
run_name="pretrain_att_maze30x30"
torchrun \
  --nnodes=$SLURM_JOB_NUM_NODES \
  --nproc_per_node=4 \
  --rdzv_id=$SLURM_JOB_ID \
  --rdzv_backend=c10d \
  --rdzv_endpoint=$MASTER_ADDR:$MASTER_PORT \
  pretrain.py \
  arch=trm \
  data_paths="[data/maze-30x30-hard-1k]" \
  evaluators="[]" \
  epochs=50000 eval_interval=5000 \
  lr=1e-4 puzzle_emb_lr=1e-4 weight_decay=1.0 puzzle_emb_weight_decay=1.0 \
  arch.L_layers=2 \
  arch.H_cycles=3 arch.L_cycles=4 \
  +run_name=${run_name} ema=True