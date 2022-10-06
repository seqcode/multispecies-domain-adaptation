#!/bin/bash
#PBS -W umask=0007
#PBS -W group_list=sam77_collab
#PBS -l walltime=10:00
#PBS -l nodes=1:ppn=8:rhel7
#PBS -j oe
#PBS -A sam77_h_g_sc_default
#PBS -l mem=100gb

cd /storage/home/vza5092/group/projects/cross-species/liver-tfs/scripts/setup_scripts

tfs="CEBPA FoxA1 HNF4a HNF6"

for tf in $tfs; do
	if [[ $tf == "CEBPA" ]]
	then
		genomes="hg38 mm10 canFam4 monDom5 galGal6 rn5 rheMac10"
	else
		genomes="hg38 mm10 canFam4 rn5 rheMac10"
	fi
	for genome in $genomes; do
		qsub -v tf=$tf,genome=$genome subsample_bound.sh
	done
done
