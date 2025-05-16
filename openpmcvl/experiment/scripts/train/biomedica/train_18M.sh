#!/bin/bash

#SBATCH --job-name=biomedica_baseline_checkpoint
#SBATCH --mem=0
# #SBATCH --qos=a40_arashaf_multimodal
# #SBATCH --partition=a40
#SBATCH --qos=deadline
#SBATCH --account=deadline
#SBATCH --partition=a100
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task=4
#SBATCH --ntasks-per-node=4
# #SBATCH --nodes=4
#SBATCH --nodes=2
#SBATCH --time=72:00:00
#SBATCH --export=ALL
#SBATCH --output=outputs/slurm-%j-%N.out
#SBATCH --open-mode=append

# load virtual environment
source /h/sajadra/pmc-data-extraction/venv/bin/activate

cd /h/sajadra/pmc-data-extraction
export PYTHONPATH="/h/sajadra/pmc-data-extraction"

export TORCH_NCCL_ASYNC_ERROR_HANDLING=3
export HYDRA_FULL_ERROR=1

export MASTER_ADDR=$(hostname)
export MASTER_PORT=45678

nvidia-smi
echo SLURM_ARRAY_JOB_ID=${SLURM_ARRAY_JOB_ID}
echo SLURM_JOBID=${SLURM_JOBID}

# “srun” executes the script <ntasks-per-node * nodes> times
srun mmlearn_run \
    'hydra.searchpath=[pkg://openpmcvl.experiment.configs]' \
    +experiment=biomedica_subfigure_18M \
    experiment_name=biomedica_subfigures_18M \
    datasets/tokenizers@dataloader.train.collate_fn.batch_processors.text=BiomedCLIPTokenizerOG \
    dataloader.train.batch_size=256 \
    dataloader.train.num_workers=4 \
    task.encoders.text.pretrained=False \
    task.encoders.rgb.pretrained=False \
    trainer.max_epochs=64 \
    task.lr_scheduler.scheduler.t_max=139597 \
    task.lr_scheduler.scheduler.warmup_length=13959 \
    trainer.num_nodes=2 \
    trainer.devices=[0,1,2,3] \
    resume_from_checkpoint=/datasets/PMC-15M/filtered_biomedica/checkpoints/second_checkpoint_18M_epoch33.ckpt


# srun mmlearn_run \
#     'hydra.searchpath=[pkg://openpmcvl.experiment.configs]' \
#     +experiment=biomedica_subfigure_18M \
#     experiment_name=biomedica_subfigures_18M \
#     datasets/tokenizers@dataloader.train.collate_fn.batch_processors.text=BiomedCLIPTokenizerOG \
#     dataloader.train.batch_size=256 \
#     dataloader.train.num_workers=4 \
#     task.encoders.text.pretrained=False \
#     task.encoders.rgb.pretrained=False \
#     trainer.max_epochs=64 \
#     task.lr_scheduler.scheduler.t_max=139597 \
#     task.lr_scheduler.scheduler.warmup_length=13959 \
#     resume_from_checkpoint=/checkpoint/sajadra/16002053/checkpoint_30.ckpt