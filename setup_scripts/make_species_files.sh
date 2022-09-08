#!/bin/bash

tf=$1

RUNS=5
EPOCHS=15

ROOT="/storage/home/vza5092/group/projects/cross-species/liver-tfs"

if [[ $tf == "CEBPA" ]]
then
	genomes="hg38 mm10 canFam4 monDom5 galGal6 rn5 rheMac10"
else
	genomes="hg38 mm10 canFam4 rn5 rheMac10"
fi

### Created files need to be as large as the largest set of bound sites 
# (of either species). So, we measure the size of all species' bound site 
# sets and go with the larger number.

bound_windows=0

for genome in $genomes
do
	bound_windows_species=`wc -l < "$ROOT/exp_data/$genome/$tf/chr3toY_pos_shuf.bed"`
	bound_windows=$((bound_windows_species>bound_windows ? bound_windows_species : bound_windows))
done

# Process of getting distinct randomly selected examples for each epoch is the
# same as in the script make_neg_window_files_for_epochs.sh.

for genome in $genomes
do
	DATA_DIR="$ROOT/exp_data/$genome/$tf"
	tmp_shuf_file="$DATA_DIR/chr3toY_shuf.tmp"  # reused each iteration
	for ((run=1;run<=RUNS;run++)); do
		shuf "$DATA_DIR/chr3toY_shuf.bed" > "$tmp_shuf_file"
		for ((epoch=1;epoch<=EPOCHS;epoch++)); do
			head_line_num=$(( bound_windows * epoch ))
			echo "For epoch $epoch, head ends at line $head_line_num"
			epoch_run_filename="$DATA_DIR/chr3toY_shuf_run${run}_${epoch}E.bed"
			head -n "$head_line_num" "$tmp_shuf_file" | tail -n "$bound_windows" > "$epoch_run_filename"
		done
	done
done

rm "$tmp_shuf_file"

echo "Done!"

exit 0

