#!/bin/bash
#PBS -W umask=0007
#PBS -W group_list=sam77_collab
#PBS -l walltime=5:00:00
#PBS -l nodes=1:ppn=1:gpus=1:rhel7
#PBS -j oe
#PBS -A sam77_i_g_gc_default
#PBS -l mem=125gb

cd $PBS_O_WORKDIR

module load cuda/10.2.89

export LD_LIBRARY_PATH="/storage/home/vza5092/usr/lib64:$LD_LIBRARY_PATH"

module load anaconda3/2020.07

source activate ~/work/tf_env


cd /storage/home/vza5092/group/projects/cross-species/liver-tfs/scripts/test_scripts

python plot_site_distribution_scatter.py "$train" "$test"