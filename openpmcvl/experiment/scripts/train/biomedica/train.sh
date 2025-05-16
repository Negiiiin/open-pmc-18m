#!/bin/bash

# add slurm config


# load virtual environment
source path/to/venv/bin/activate

cd path/to/root
export PYTHONPATH=path/to/root

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
    +experiment=biomedica_matched \
    experiment_name=biomedica_baseline_checkpoint \
    datasets/tokenizers@dataloader.train.collate_fn.batch_processors.text=BiomedCLIPTokenizerOG \
    dataloader.train.batch_size=256 \
    dataloader.train.num_workers=4 \
    task.encoders.text.pretrained=False \
    task.encoders.rgb.pretrained=False \
    trainer.max_epochs=64 \
    task.lr_scheduler.scheduler.t_max=48436 \
    task.lr_scheduler.scheduler.warmup_length=4843 \
    trainer.num_nodes=2 \
    trainer.devices=[0,1,2,3] \
    resume_from_checkpoint=None
