#!/bin/bash
#PBS -W umask=0007
#PBS -W group_list=sam77_collab
#PBS -l walltime=50:00:00
#PBS -j oe
#PBS -A sam77_h_g_sc_default
#PBS -l mem=100gb
#PBS -l nodes=1:ppn=8


# expects arg $genome


case "$genome" in
  mm10) genome_long="Mus musculus;mm10";;
  hg38) genome_long="Homo sapiens;hg38";;
  canFam4) genome_long="Canis familiaris;canFam4";;
  galGal6) genome_long="Gallus gallus;galGal6";;
  monDom5) genome_long="Monodelphis domestica;monDom5";;
  rheMac10) genome_long="Macaca mulatta;rheMac10";;
  rn5) genome_long="Rattus norvegicus;rn5";;
  *)
    echo "Error: genome not recognized."
    exit 1
  ;;
esac

cd /gpfs/group/sam77/default/projects/cross-species/liver-tfs/exp_data

echo "Running DomainFinder..."

java -Xmx30G -cp /gpfs/group/sam77/default/code/jars/multigps.v0.75.mahonylab.jar org.seqcode.projects.seed.DomainFinder --threads 4 --species "${genome_long}" --seq "/gpfs/group/sam77/default/genomes/$genome/" --design "$genome/blacklist_dc.design" --binwidth 50 --binstep 25 --mergewin 200 --poisslogpthres -5 --binpthres 0.01 --out "$genome/info/domains" > $genome/info/domainFinder.out  2>&1

echo "Done!"
