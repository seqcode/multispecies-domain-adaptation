import sys
import argparse

def make_windows(gInfo, windowSize, windowStride, outFileName = None):
	outFileName = "/storage/home/vza5092/group/projects/cross-species/liver-tfs/exp_data/" + gInfo.split(".")[0] + "/info/" + outFileName
	with open(gInfo, "r") as gInfoFile, open(outFileName, "w") as outFile:
		for chromLine in gInfoFile:
			chrom,length = chromLine.strip().split()
			length = int(length)
			index = 0
			while index + windowSize < length:
				outFile.write("\t".join([chrom, str(index), str(index + windowSize) + "\n"]))
				index += windowStride

if __name__ == "__main__":
	parser = argparse.ArgumentParser(description="Use the genome.info file to create discrete windows on every chromosome, in bed file format. Binding events can be mapped to the output windows file.")
	parser.add_argument("windowSize", type=int, help="use 500 as default", default=500)
	parser.add_argument("windowStride", type=int, help="use 50 as default", default=50)
	parser.add_argument("genomeInfo", type=str, default="mm10.info")
	parser.add_argument("--out", type=str, dest="outFileName", default="windows.bed")
	args = parser.parse_args()

	make_windows(args.genomeInfo, args.windowSize, args.windowStride, args.outFileName)
