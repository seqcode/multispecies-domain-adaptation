#!/bin/bash
#PBS -W umask=0007
#PBS -W group_list=sam77_collab
#PBS -l walltime=10:00
#PBS -l nodes=1:ppn=8:rhel7
#PBS -j oe
#PBS -A sam77_h_g_sc_default
#PBS -l mem=50gb


cd /storage/home/vza5092/group/projects/cross-species/liver-tfs/scripts/model_training

tfs="CEBPA FoxA1 HNF4a HNF6"
genomes="hg38"

# runs=( 1 2 3 4 5 )

runs=( 1 )

for tf in $tfs
do

	for genome in $genomes
	do
		for run in ${runs[@]}; do
			qsub -v tf=$tf,genome=$genome,run=$run run_reg_training.sh
		done
	done
done