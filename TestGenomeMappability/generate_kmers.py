import sys
import argparse


def generate_kmers(gInfo, gDir, k):
	outFile = "../../../../seqalign/reads/mappability/genomes/" + gDir.split("/")[-2] + ".k" + str(k) + ".fq"
	with open(gInfo, "r") as genomeInfo, open(outFile, "w") as outFile:
		kmer_index = 0
		genome_index = 0
		for chromLine in genomeInfo:
			chrom, length = chromLine.strip().split()
			length = int(length)
			prevLine = ""
			with open(gDir + "/" + chrom + ".fa", "r") as gen:
				next(gen)
				for genLine in gen:
					genLine = genLine.strip()
					currLine = prevLine[-k+1:] + genLine
					for i in range(len(currLine) - k + 1):
						kmer = currLine[i:i + k]
						if "N" not in kmer:
							outFile.write("@kmer" + str(kmer_index) + "\n")
							outFile.write(kmer + "\n")
							outFile.write("+" + "\n")
							outFile.write("I"*k + "\n")
							kmer_index += 1
					prevLine = genLine



if __name__ == "__main__":
	parser = argparse.ArgumentParser(description="Use the reference genome file and the genome info file to generate all k-mers of a given length to map against genome.")
	parser.add_argument("genomeInfo", type=str, help="genome info file", default="mm10.info")
	parser.add_argument("genomeDir", type=str, help="directory where genome fasta files are located", default="~/group/genomes/mm10")
	parser.add_argument("k", type=int, default=36)
	args = parser.parse_args()

	generate_kmers(args.genomeInfo, args.genomeDir, args.k)
