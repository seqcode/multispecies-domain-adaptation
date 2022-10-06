#!/bin/bash
#PBS -W umask=0007
#PBS -W group_list=sam77_collab
#PBS -l walltime=50:00:00
#PBS -l nodes=1:ppn=8:rhel7
#PBS -j oe
#PBS -A sam77_h_g_sc_default
#PBS -l mem=100gb


tf=$1
genome=$2


windowSize=500

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

# make tf labels

bindingIntersectFile="binding_${windowSize}_intersect"
sortedIntersectFile="${bindingIntersectFile}.dictsort.bed"

cd "/storage/home/vza5092/group/projects/cross-species/liver-tfs/exp_data/${genome}"

mkdir -p "${tf}"

bedtools intersect -a "info/windows_good.bed" -b "${name}-${tf}/${name}-${tf}_${tf}.bed" -wa > "${tf}/${bindingIntersectFile}.bed"

cd "${tf}"

sort "${bindingIntersectFile}.bed" | uniq > "${sortedIntersectFile}"

rm "${bindingIntersectFile}.bed"

# assign 1 labels to windows that intersected peaks
sed "s/$/	1/" "${sortedIntersectFile}" > "TFlabels.unsorted.bed.pos"

if [[ ! -f "../info/windows_good.dictsort.bed" ]]; then
  sort "../info/windows_good.bed" > "../info/windows_good.dictsort.bed"
fi

# assign 0 labels to all other windows
comm "../info/windows_good.dictsort.bed" "${sortedIntersectFile}" -23 | sed "s/$/	0/" > "TFlabels.unsorted.bed.neg"

# merge the two and sort
cat "TFlabels.unsorted.bed.neg" "TFlabels.unsorted.bed.pos" | sort -k1,1 -k2,2n > "TFlabels.bed"


