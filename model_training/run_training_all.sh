#!/bin/bash
#PBS -W umask=0007
#PBS -W group_list=sam77_collab
#PBS -l walltime=10:00:00
#PBS -l nodes=1:ppn=8:rhel7
#PBS -j oe
#PBS -A sam77_h_g_sc_default
#PBS -l mem=50gb


cd /storage/home/vza5092/group/projects/cross-species/liver-tfs/scripts/model_training

tfs="CEBPA FoxA1 HNF4a HNF6"

genomes="hg38 mm10 canFam4 rn5 rheMac10 monDom5 galGal6"

genomes1="hg38 mm10 canFam4 rn5 rheMac10"

genomes3="hg38"


runs=( 1 )


for run in ${runs[@]}; do
	for genome in $genomes3
	do
		tfs="CEBPA FoxA1 HNF4a HNF6"

		for tf in $tfs
		do
			qsub -v tf=$tf,genome=$genome,run=$run run_training.sh
			qsub -v tf=$tf,genome=$genome,run=$run run_DA_training.sh
			qsub -v tf=$tf,genome=$genome,run=$run run_BA_training.sh
			qsub -v tf=$tf,genome=$genome,run=$run run_multi_training.sh
		done
	done
done