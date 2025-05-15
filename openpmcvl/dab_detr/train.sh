#!/bin/bash

# Add sbatch configs

set -e  # Exit script on error

echo "Running on $(hostname)"
echo "Job started at $(date)"

export PROJECT_ROOT=#path/to/directory containing train.py
source #path/to/venv/bin/activate  # Activate virtual environment

cd $PROJECT_ROOT
export PYTHONPATH=$PROJECT_ROOT

# Log system info
echo "Environment Variables:"
env | grep SLURM
echo "CUDA Info:"
nvidia-smi

echo "Script received arguments: $@"

# Run training with arguments if provided
python train.py "$@"  

echo "Job finished at $(date)"