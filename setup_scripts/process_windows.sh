#!/bin/bash
#PBS -W umask=0007
#PBS -W group_list=sam77_collab
#PBS -l walltime=50:00:00
#PBS -l nodes=1:ppn=8:rhel7
#PBS -j oe
#PBS -A sam77_h_g_sc_default
#PBS -l mem=100gb


cd /storage/home/vza5092/group/projects/cross-species/liver-tfs/exp_data/hg38/info

genomes="canFam4 monDom5 galGal6 rn5 rheMac10"
for genome in $genomes; do
	cd ../../$genome/info

	# remove windows that intersect blacklist regions
	bedtools intersect -v -a "windows.bed" -b "domains/domains_ChIPseq.p0.01.bed" > "windows_noBL.bed"
	# keep only windows that intersect mappable regions
	
	mappability="/storage/home/vza5092/group/genomes/$genome/mappability"

	cat "$mappability/${genome}.genmap.bedgraph" | awk '{if($4<1) print $1 "\t" $2 "\t" $3}' > unmappable.bed

	bedtools intersect -v -a "windows_noBL.bed" -b "unmappable.bed" > "windows_good.bed" 

done

genomes="hg38 mm10"
for genome in $genomes; do
	cd ../../$genome/info

	# remove windows that intersect blacklist regions
	bedtools intersect -v -a "windows.bed" -b "blacklist_regions.bed" > "windows_noBL.bed"
	
	# remove windows that intersect unmappable regions
	bedtools intersect -v -a "windows_noBL.bed" -b "lt0.8umap.windows.bed" > "windows_good.bed"

done
