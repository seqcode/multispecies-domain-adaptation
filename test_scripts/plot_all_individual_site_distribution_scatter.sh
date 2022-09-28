#!/bin/bash
#PBS -W umask=0007
#PBS -W group_list=sam77_collab
#PBS -l walltime=5:00:00
#PBS -l nodes=1:ppn=8:rhel7
#PBS -j oe
#PBS -A sam77_h_g_sc_default
#PBS -l mem=50gb

cd $PBS_O_WORKDIR

module load cuda/10.2.89

export LD_LIBRARY_PATH="/storage/home/vza5092/usr/lib64:$LD_LIBRARY_PATH"

module load anaconda3/2020.07

source activate ~/work/tf_env


cd /storage/home/vza5092/group/projects/cross-species/liver-tfs/scripts/test_scripts

train_genomes="mm10 mm10_DA canFam4 canFam4_DA monDom5 monDom5_DA galGal6 galGal6_DA rn5 rn5_DA rheMac10 rheMac10_DA"
for train_genome in $train_genomes
do
	python3 plot_individual_site_distribution_scatter.py $train_genome "hg38"
done

