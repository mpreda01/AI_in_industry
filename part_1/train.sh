#!/bin/bash
#SBATCH --job-name=train
#SBATCH --mail-type=ALL
#SBATCH --mail-user=matteo.preda2@studio.unibo.it
#SBATCH --time=12:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=31G
#SBATCH --partition=rtx2080
#SBATCH --gres=gpu:1
#SBATCH --output=train_%j.log
#SBATCH --error=train_%j.err
#SBATCH --chdir=/scratch.hpc/matteo.preda

# Usage (from the repository root): sbatch train.sh [section.key=value ...]
# The run is defined by config.yaml; extra arguments override single values, e.g. wandb.mode=offline

export DATASET_ROOT=/scratch.hpc/matteo.preda/industry/synscapes/synscapes_processed
export SAVEDIR=/scratch.hpc/matteo.preda/industry/runs
VENV_DIR=/scratch.hpc/matteo.preda/industry/indus/venv
# The 400 MB home quota is full: keep matplotlib/torch caches on scratch.
export XDG_CACHE_HOME=/scratch.hpc/matteo.preda/.cache

cd "${SLURM_SUBMIT_DIR}"

"${VENV_DIR}/bin/python3" -u -m source.scripts.train --config config.yaml \
  data.workers="${SLURM_CPUS_PER_TASK}" "$@"
