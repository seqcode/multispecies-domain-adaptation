#!/bin/bash
#PBS -W umask=0007
#PBS -W group_list=sam77_collab
#PBS -l walltime=90:00:00
#PBS -l nodes=1:ppn=1:gpus=1:rhel7
#PBS -j oe
#PBS -A sam77_i_g_gc_default
#PBS -l mem=125gb


cd $PBS_O_WORKDIR

module load cuda/10.2.89

export LD_LIBRARY_PATH="/storage/home/vza5092/usr/lib64:$LD_LIBRARY_PATH"

module load anaconda3/2020.07

source activate ~/work/tf_env



cd /storage/home/vza5092/group/projects/cross-species/liver-tfs/scripts/model_training

ulimit -n 4096


LOG_ROOT="/storage/home/vza5092/group/projects/cross-species/liver-tfs/logs/training"

parent_dir="/storage/home/vza5092/group/projects/cross-species/liver-tfs/models"


mkdir -p "$parent_dir/$tf/${genome}_trained/basic_model"
mkdir -p "$parent_dir/$tf/${genome}_trained/multi_model"
mkdir -p "$parent_dir/$tf/${genome}_trained/DA_model"
mkdir -p "$parent_dir/$tf/${genome}_trained/BA_model"

python BA_train.py "$tf" "$genome" "$run" > "$LOG_ROOT/BA_${genome}_${tf}_run${run}.log"