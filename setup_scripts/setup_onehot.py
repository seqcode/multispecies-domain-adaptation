import sys
import argparse
import os
from seqdataloader.batchproducers.coordbased.core import Coordinates
from seqdataloader.batchproducers.coordbased.coordstovals.fasta import PyfaidxCoordsToVals
import pickle

GENOMES = {"mm10" : "/storage/home/vza5092/group/genomes/mm10/mm10.fa",
       "hg38" : "/storage/home/vza5092/group/genomes/hg38/hg38.fa",
       "monDom5" : "/storage/home/vza5092/group/genomes/monDom5/monDom5.fa",
       "canFam4" : "/storage/home/vza5092/group/genomes/canFam4/canFam4.fa",
       "galGal6" : "/storage/home/vza5092/group/genomes/galGal6/galGal6.fa",
       "rn5" : "/storage/home/vza5092/group/genomes/rn5/rn5.fa",
       "rheMac10" : "/storage/home/vza5092/group/genomes/rheMac10/rheMac10.fa"}

ROOT="/storage/home/vza5092/group/projects/cross-species/liver-tfs/exp_data/"


def make_onehot(species, tf):
	LOG_ROOT="/storage/home/vza5092/group/projects/cross-species/liver-tfs/scripts/setup_scripts"

	log_fn = LOG_ROOT+"/SOH_"+species+"_"+tf+".log"
	try:
		search_dir = ROOT+species+"/"+tf
		all_files = [fn for fn in os.listdir(search_dir) if "run" in fn]
		unwanted = ["chr2.bed", "chr3toY_neg_shuf.bed", "chr3toY_shuf.bed", "chr3toY_shuf.tmp"]
		filenames = [fn for fn in all_files if (fn not in unwanted and fn.startswith('chr'))]
		species_converter = PyfaidxCoordsToVals(GENOMES[species])
		for fn in filenames:
			with open(search_dir+"/"+fn) as f:
				coords_tmp = [line.split()[:3] for line in f]
				coords = [Coordinates(coord[0], int(coord[1]), int(coord[2])) for coord in coords_tmp]
				coords_onehot = species_converter(coords)
			new_fn = fn.split(".")
			new_fn = new_fn[0] + "_onehot." + new_fn[1]
			with open(search_dir+"/"+new_fn, 'wb') as f:
				pickle.dump(coords_onehot, f)
				with open(log_fn, "w") as log_file:
					log_file.write(new_fn + "\n")
	except Exception as e:
		with open(log_fn, "w") as log_file:
			log_file.write(str(e) + "\n")
		raise e




if __name__ == "__main__":
	parser = argparse.ArgumentParser(description="Use the genome.info file to create discrete windows on every chromosome, in bed file format. Binding events can be mapped to the output windows file.")
	parser.add_argument("species", type=str, help="use 500 as default", default=500)
	parser.add_argument("tf", type=str, help="use 50 as default", default=50)
	# parser.add_argument("genomeInfo", type=str, default="mm10.info")
	# parser.add_argument("--out", type=str, dest="outFileName", default="windows.bed")
	args = parser.parse_args()

	make_onehot(args.species, args.tf)