#!/bin/bash

tf=$1
genome=$2

ROOT="/storage/home/vza5092/group/projects/cross-species/liver-tfs"

DATA_DIR="$ROOT/exp_data/$genome/$tf"

### Make chr1/2 validation and test sets
# This script creates a validation set of 1 million randomly sampled
# (without replacement) windows from chromosome 1, and a test set of all
# windows on chromosome 2.

allbed="$DATA_DIR/TFlabels.bed"

chr1file="$DATA_DIR/chr1_random_1m.bed"
chr2file="$DATA_DIR/chr2.bed"

grep -F "chr1	" "$allbed" | shuf | head -n1000000 > "$chr1file"
grep -F "chr2	" "$allbed" > "$chr2file"

# Sanity checks

chr1_windows=`wc -l < "$chr1file" `
# if [[ "$chr1_windows" != 1000000 ]]; then
# 	echo "Error: chr1 val set file only has $chr1_windows windows. Exiting."
# 	exit 1
# fi

chr1_chroms=`awk '{ print $1 }' "$chr1file" | sort | uniq | wc -l `
if [[ "$chr1_chroms" != 1 ]]; then
	echo "Error: chr1 val set file contains mutliple chromosomes. Exiting."
	exit 1
fi

chr2_chroms=`awk '{ print $1 }' "$chr2file" | sort | uniq | wc -l `
if [[ "$chr2_chroms" != 1 ]]; then
	echo "Error: chr2 test set file contains mutliple chromosomes. Exiting."
	exit 1
fi

### Get training chromosomes, split into bound/unbound examples
# Here we divide the training examples (examples from chromosomes except 1, 2)
# into bound and unbound examples. Next we will sample balanced
# (half bound, half unbound) training datasets from these files.

grep -Ev "chr[12]	" "$allbed" | shuf > "$DATA_DIR/chr3toY_shuf.bed"
awk '$NF == 1' "$DATA_DIR/chr3toY_shuf.bed" | shuf > "$DATA_DIR/chr3toY_pos_shuf.bed"
awk '$NF == 0' "$DATA_DIR/chr3toY_shuf.bed" | shuf > "$DATA_DIR/chr3toY_neg_shuf.bed"

total_windows=`wc -l < "$DATA_DIR/chr3toY_shuf.bed"`
bound_windows=`wc -l < "$DATA_DIR/chr3toY_pos_shuf.bed"`
unbound_windows=`wc -l < "$DATA_DIR/chr3toY_neg_shuf.bed"`

total=$(( $bound_windows + $unbound_windows ))
if [[ $total != $total_windows ]]; then
	echo "Error: bound + unbound windows does not equal total windows. Exiting."
	exit 1
fi

echo "Bound training windows: $bound_windows"
echo "Unbound training windows: $unbound_windows"

let possible=${unbound_windows}/$bound_windows

EPOCHS=15

if [ $possible -lt $EPOCHS ]; then

	let need_fifteen=${unbound_windows}/${EPOCHS}
	echo "Per epoch: Want $need_fifteen"

	let small=${need_fifteen}/5

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

	GEN_DIR="$ROOT/exp_data/$genome"

	head -n $small "$GEN_DIR/${name}-${tf}/${name}-${tf}_${tf}.bed" > "$DATA_DIR/${genome}-${tf}_${tf}_15.bed"

	bedtools intersect -a "$DATA_DIR/chr3toY_pos_shuf.bed" -b "$DATA_DIR/${genome}-${tf}_${tf}_15.bed" -wa > "$DATA_DIR/chr3toY_pos_shuf_15.bed"

	top=`wc -l < "$DATA_DIR/chr3toY_pos_shuf_15.bed"`
	echo "Top binding sites 2: $top"

	head -n $need_fifteen "$DATA_DIR/chr3toY_pos_shuf_15.bed" | shuf > "$DATA_DIR/chr3toY_pos_shuf.bed"

	bound_windows=`wc -l < "$DATA_DIR/chr3toY_pos_shuf.bed"`

	echo "Bound training windows: $bound_windows"

fi


### Shuffle the unbound sites that our training data can sample from, and then
# sample a distinct set of sites for training each epoch (no unbound site is
# repeated across multuple epochs). The bound sites trained on each epoch are
# the same. The number of unbound sites sampled each epoch is equal to the
# number of bound sites to be used in training (so each epoch's training data
# will be balanced).

# This process will create a file of unbound sites for training specific to
# each epoch. If you opt for too many epochs (or the size of your training 
# datasets is too large), this script may error out when it uses up all 
# possible unbound sites to sample from.

# RUNS: the number of replicate model training runs to make data for.
RUNS=5
# EPOCHS: the number of epochs the models will train for.
# Files will be generated for each epoch.

# since we want new unbound windows in every epoch the max number of epochs
# is unbound / bound

# possible=`expr ${unbound_windows} / $bound_windows`

# that might be a lot so I set an upper bound.
# just making sure that we don't run out of unbound windows.

# upper=15

# EPOCHS=$(($possible<$upper ? $possible : $upper))

POS_FILE="$DATA_DIR/chr3toY_pos_shuf.bed"
NEG_FILE="$DATA_DIR/chr3toY_neg_shuf.bed"

tmp_shuf_file="$DATA_DIR/chr3toY_neg_shuf.tmp"  # reused each iteration
for ((run=1;run<=RUNS;run++)); do
	shuf "$NEG_FILE" > "$tmp_shuf_file"
	for ((epoch=1;epoch<=EPOCHS;epoch++)); do
		head_line_num=$(( bound_windows * epoch ))
		echo "For epoch $epoch, head ends at line $head_line_num"
		epoch_run_filename="$DATA_DIR/chr3toY_neg_shuf_run${run}_${epoch}E.bed"
		head -n "$head_line_num" "$tmp_shuf_file" | tail -n "$bound_windows" > "$epoch_run_filename"

		# sanity check
		lines_in_file=`wc -l < "$epoch_run_filename"`
		echo "Lines in $epoch_run_filename: $lines_in_file"
		if [[ "$lines_in_file" != "$bound_windows" ]]; then
			echo "Error: incorrect number of lines ($lines_in_file) in file $epoch_run_filename (should be $bound_windows). Exiting."
			exit 1
		fi
	done
done

rm "$tmp_shuf_file"


echo "Done!"

exit 0
