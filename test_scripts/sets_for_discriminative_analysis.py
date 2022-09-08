import numpy as np
from collections import defaultdict
import sys

ROOT = "/storage/home/vza5092/group/projects/cross-species/liver-tfs/"
RESULT_ROOT = ROOT + "model_out/discriminative_analysis/"

SPECIES = ["mm10", "hg38", "canFam4", "rn5", "rheMac10", "monDom5", "galGal6"]
SPECIES_SMALL = ["mm10", "hg38", "canFam4", "rn5", "rheMac10"]

# model_types = ["DA", "BA", "Multi"]
model_type_names = ["DA_model", "BA_model", "multi_model"]

test_species = sys.argv[1]
if test_species in SPECIES_SMALL:
    TFS = ["CEBPA", "FoxA1", "HNF4a", "HNF6"]
else:
    TFS = ["CEBPA"]

def get_preds_file(tf, train_species, test_species, model_type):
    preds_root = ROOT + "/model_out/"
    return(preds_root + tf + "_" + model_type + "_" + train_species + "-trained_" + test_species + "-test.preds.npy")

def get_test_file(test_species):
	return(ROOT + "exp_data/" + test_species + "/CEBPA/chr2.bed")

def get_matched_result_file(tf, model_type, test_species):
	return(RESULT_ROOT + tf+"_"+test_species+"_"+model_type+"_bound.bed")

def get_unmatched_result_file(tf, model_type, test_species):
	return(RESULT_ROOT + tf+"_"+test_species+"_bound_"+model_type+"_unbound.bed")

for tf in TFS:
	for model_type in model_type_names:
		singlespecies_preds = get_preds_file(tf, test_species, test_species, "basic_model")
		singlespecies_preds = np.load(singlespecies_preds)

		multispecies_preds = get_preds_file(tf, "all-but-"+test_species, test_species, model_type)
		multispecies_preds = np.load(multispecies_preds)

		test_bed = get_test_file(test_species)
		f = open(test_bed, "r")

		result_matched = get_matched_result_file(tf, model_type, test_species)
		f_matched = open(result_matched, "a")

		result_unmatched = get_unmatched_result_file(tf, model_type, test_species)
		f_unmatched = open(result_unmatched, "a")

		for i in range(min(len(singlespecies_preds), len(multispecies_preds))):
			line = f.readline()
			if line.split()[-1] == "1":
				if(multispecies_preds[i] > 0.5 and singlespecies_preds[i] > 0.5):
					f_matched.write(line)
				elif(multispecies_preds[i] < 0.5 and singlespecies_preds[i] > 0.5):
					f_unmatched.write(line)