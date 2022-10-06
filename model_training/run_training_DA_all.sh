#!/bin/bash
#PBS -W umask=0007
#PBS -W group_list=sam77_collab
#PBS -l walltime=10:00
#PBS -l nodes=1:ppn=4:rhel7
#PBS -j oe
#PBS -A sam77_h_g_sc_default
#PBS -l mem=50gb


cd /storage/home/vza5092/group/projects/cross-species/liver-tfs/scripts/model_training

tfs="CEBPA FoxA1 HNF4a HNF6"

# runs=( 1 2 3 4 5 )

runs=( 1 )

for tf in $tfs
do
	if [[ $tf == "CEBPA" ]]
	then
		genomes="hg38 mm10 canFam4 monDom5 galGal6 rn5 rheMac10"
	else
		genomes="hg38 mm10 canFam4 rn5 rheMac10"
	fi

	for genome in $genomes
	do
		for run in ${runs[@]}; do
			qsub -v tf=$tf,genome=$genome,run=$run run_DA_training.sh
		done
	done
done