#!/bin/bash

# Define the partition on which the job shall run.
#SBATCH --partition xxx

# Define a name for your job
#SBATCH --job-name AutoGluon_Pipeline

# Define the files to write the outputs of the job to.
#SBATCH --output logs/%x-%A_%a.out
#SBATCH --error logs/%x-%A_%a.err

# Define the amount of memory required per node
#SBATCH --mem 32GB
#SBATCH --cpus-per-task=8
#SBATCH --gres=localtmp:100

#SBATCH --propagate=NONE

directory_path="src/datasets/feature_engineered_datasets/"

# shellcheck disable=SC2034
dir_count=$(find "$directory_path" -mindepth 1 -maxdepth 1 -type d | wc -l)
#SBATCH --array=0-$((dir_count-1))

echo "Workingdir: $PWD"
echo "Started at $(date)"

echo "Running job $SLURM_JOB_NAME using $SLURM_JOB_CPUS_PER_NODE cpus per node with given JID $SLURM_JOB_ID on queue $SLURM_JOB_PARTITION"

# shellcheck disable=SC1090
source ~/miniconda3/bin/activate
conda activate amltk_env
echo "conda amltk_env activated"

python3 -m pip install --upgrade pip
pip install -r requirements.txt
echo "Requirements installed"
# shellcheck disable=SC1068

# Set the PYTHONPATH to include the src directory
export PYTHONPATH=$PWD/src:$PYTHONPATH
echo "PYTHONPATH set to $PYTHONPATH"


# shellcheck disable=SC2207
methods=($(find "$directory_path" -mindepth 1 -maxdepth 1 -type d))
method="${methods[$SLURM_ARRAY_TASK_ID]}"

# shellcheck disable=SC2006
start=`date +%s`

echo "Running Method: $method"
python3 src/autogluon/run_autogluon_parallel.py --method "$method"

# shellcheck disable=SC2006
end=`date +%s`
runtime=$((end-start))

echo "Job execution complete."
echo "Runtime: $runtime"