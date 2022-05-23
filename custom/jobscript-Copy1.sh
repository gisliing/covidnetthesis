#!/usr/bin/env bash

#slurm job configuration
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1
#SBATCH --output=outputs/output_%j.out
#SBATCH --error=errors/error_%j.er
#SBATCH --time=6:00:00
#SBATCH --job-name=covid19
#SBATCH --gres=gpu:1 --partition=ml-gpu
#SBATCH --account joaiml

module -q purge
module -q use $OTHERSTAGES
module -q load Stages/Devel-2019a GCC/8.3.0 GCCcore/.8.3.0 ParaStationMPI/5.4.0-1-CUDA
module -q load Horovod/0.16.2-GPU-Python-3.6.8
module -q load TensorFlow/1.13.1-GPU-Python-3.6.8
module list

source /p/project/joaiml/ingolfsson1/jupyter/kernels/covid_kernel/bin/activate

export PYTHONPATH=/p/project/joaiml/ingolfsson1/jupyter/kernels/covid_kernel/lib/python3.6/site-packages:${PYTHONPATH}
          
# Run the program
srun --cpu-bind=none,v --accel-bind=gn python -u custom_inference_deep.py
