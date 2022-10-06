#!/bin/bash
#PBS -W umask=0007
#PBS -W group_list=sam77_collab
#PBS -l walltime=50:00:00
#PBS -l nodes=1:ppn=8:rhel7
#PBS -j oe
#PBS -A sam77_h_g_sc_default
#PBS -l mem=100gb


windowSize=500

# need to pass in tf and genome as variables tf and genome


destDir="/storage/home/vza5092/group/projects/cross-species/liver-tfs/exp_data/${genome}"

case "$genome" in
  mm10) name="mouse";;
  hg38) name="human";;
  canFam4) name="dog";;
  galGal6) name="chicken";;
  monDom5) name="possum";;
  rheMac10) name="macaque";;
  rn5) name="rat";;
  *)
    echo "Error: genome not recognized."
    exit 1
  ;;
esac

cd "/storage/home/vza5092/group/projects/cross-species/liver-tfs/exp_data/${genome}"

head -n 5000 "${name}-${tf}/${name}-${tf}_${tf}.bed" > "${tf}/${genome}-${tf}_${tf}_small.bed"

cd "${tf}"

# bedtools intersect -a "TFlabels.unsorted.bed.pos" -b "${genome}-${tf}_${tf}_small.bed" -wa > "TFlabels.unsorted.bed.pos.small"

bedtools intersect -a "chr3toY_pos_shuf.bed" -b "${genome}-${tf}_${tf}_small.bed" -wa > "chr3toY_pos_shuf_small.bed"


