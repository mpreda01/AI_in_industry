#!/bin/bash
#SBATCH --job-name=preprocess_synscapes
#SBATCH --mail-type=ALL
#SBATCH --mail-user=matteo.preda2@studio.unibo.it
#SBATCH --time=08:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=31G
#SBATCH --partition=rtx2080
#SBATCH --output=preprocess_synscapes_%j.log
#SBATCH --error=preprocess_synscapes_%j.err
#SBATCH --chdir=/scratch.hpc/matteo.preda
# GPU note: this job is CPU/IO-bound (EXR decode + LUT lookup + jpeg encode) and
# never touches the GPU. Confirmed with the user that this cluster requires
# --gres=gpu:1 on every job regardless of use, so it is kept (each node has a
# single GPU, this is the only value that works). Partition is l40 rather than
# rtx2080 purely for its 8 CPU cores vs 4, since --workers below parallelizes
# on CPU count and we're allocating a GPU node either way.
#SBATCH --gres=gpu:1

# --- Fill in before submitting ------------------------------------------------------
# Python virtual environment on this cluster, created on giano and populated with
# `pip3 install -r requirements.txt --no-cache-dir` (see the cluster instructions:
# pip's default cache lives in the 400 MB home quota and fills it fast).
VENV_DIR=/scratch.hpc/matteo.preda/industry/indus/venv

# Absolute path to the ORIGINAL raw Synscapes root (contains img/ and meta/).
# TODO: fill this in - it cannot be guessed.
SYNSCAPES_ROOT=/scratch.hpc/matteo.preda/industry/synscapes/Synscapes

# Where the Miniscapes-formatted output is written. This MUST be under
# /scratch.hpc/ (or another quota-exempt path): the 400 MB home quota cannot
# hold 25000 images across rgb/semseg/depth (order of tens of GB).
# IMPORTANT: verify /scratch.hpc/ is actually visible from the compute node this
# job lands on before trusting this path - see the note in chat about the
# cluster instructions being self-contradictory on this point. If it turns out
# /scratch.hpc/ is NOT visible from compute nodes, this whole approach needs a
# different output location (ask IT, or route through the shared home instead).
OUTPUT_ROOT=/scratch.hpc/matteo.preda/industry/synscapes/synscapes_processed

# Split seed: fixed and logged (see preprocess_manifest.json in OUTPUT_ROOT after
# the run). Change only if you deliberately want a different train/val/test split.
SPLIT_SEED=42
# --------------------------------------------------------------------------------

# Locate this script's own directory (works regardless of --chdir), same pattern
# already used by slurm_train.sh in this repo.
SCRIPTDIR=$(scontrol show job "$SLURM_JOB_ID" | awk -F= '/Command=/{print $2}')
SCRIPTDIR=$(realpath "$SCRIPTDIR")
cd "$(dirname "$SCRIPTDIR")"

echo "Preprocessing Synscapes -> Miniscapes"
echo "  synscapes_root=${SYNSCAPES_ROOT}"
echo "  output_root=${OUTPUT_ROOT}"
echo "  split_seed=${SPLIT_SEED}"
echo "  cpus=${SLURM_CPUS_PER_TASK}"

"${VENV_DIR}/bin/python3" preprocess_synscapes.py \
  --synscapes-root "${SYNSCAPES_ROOT}" \
  --output-root "${OUTPUT_ROOT}" \
  --rgb-source rgb \
  --split-seed "${SPLIT_SEED}" \
  --workers "${SLURM_CPUS_PER_TASK}"
