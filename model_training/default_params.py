from math import floor
from subprocess import check_output
from pprint import pprint
import numpy as np
from datetime import datetime


SPECIES = ["mm10", "hg38", "monDom5", "canFam4", "galGal6", "rn5", "rheMac10"]
SPECIES_SMALL = ["mm10", "hg38", "canFam4", "rn5", "rheMac10"]

# Need to provide seqdataloader with locations of genome fasta files
# seqdataloader is expecting that there will be a fasta index in the same directory
GENOMES = {"mm10" : "/storage/home/vza5092/group/genomes/mm10/mm10.fa",
       "hg38" : "/storage/home/vza5092/group/genomes/hg38/hg38.fa",
       "monDom5" : "/storage/home/vza5092/group/genomes/monDom5/monDom5.fa",
       "canFam4" : "/storage/home/vza5092/group/genomes/canFam4/canFam4.fa",
       "galGal6" : "/storage/home/vza5092/group/genomes/galGal6/galGal6.fa",
       "rn5" : "/storage/home/vza5092/group/genomes/rn5/rn5.fa",
       "rheMac10" : "/storage/home/vza5092/group/genomes/rheMac10/rheMac10.fa"}


TFS = ["CEBPA", "FoxA1", "HNF4a", "HNF6"]
DATA_ROOT = "/storage/home/vza5092/group/projects/cross-species/liver-tfs/exp_data/"

# These files are created by the script runall_setup_training_data.sh
VAL_FILENAME = "chr1_random_1m.bed"
TRAIN_POS_FILENAME = "chr3toY_pos_shuf.bed"
TRAIN_NEG_FILENAME = "chr3toY_neg_shuf_runX_1E.bed"

# where models will be saved during/after training
MODEL_ROOT = "/storage/home/vza5092/group/projects/cross-species/liver-tfs/models/"
TIMESTAMP_FORMAT = "%Y-%m-%d_%H-%M-%S"


class Params:
    def __init__(self, args):
        self.batchsize = 400
        self.seqlen = 500
        self.convfilters = 240
        self.filtersize = 20
        self.strides = 15
        self.pool_size = 15
        self.lstmnodes = 32
        self.dl1nodes = 1024
        self.dl2nodes = 512
        self.dropout = 0.5
        self.valbatchsize = 1000
        self.epochs = 15

        self.parse_args(args)
        self.set_steps_per_epoch()
        self.chromsize = 0  # leftover from when accessibility was also input to model

        pprint(vars(self))

        self.set_val_labels()


    def parse_args(self, args):
        # NOTE: this method is expecting arguments input in a particular order!!
        assert len(args) >= 4, len(args)
        self.tf = args[1]
        assert self.tf in TFS, self.tf
        self.source_species = args[2]
        assert self.source_species in SPECIES, self.source_species  
        self.run = int(args[3])
        
        source_root = DATA_ROOT + self.source_species + "/" + self.tf + "/"
        if(self.tf == 'CEBPA'):
            self.target_species = [species for species in SPECIES if species != self.source_species]
            self.all_species = SPECIES
        else:
            self.target_species = [species for species in SPECIES_SMALL if species != self.source_species]
            self.all_species = SPECIES_SMALL
        target_root = [DATA_ROOT + target + "/" + self.tf + "/" for target in self.target_species]

        self.bindingtrainposfile = source_root + TRAIN_POS_FILENAME
        self.bindingtrainnegfile = source_root + TRAIN_NEG_FILENAME
        self.bindingtrainnegfile = self.bindingtrainnegfile.replace("runX", "run" + str(self.run))
        
        self.sourcevalfile = source_root + VAL_FILENAME
        self.targetvalfile = [target + VAL_FILENAME for target in target_root]

        timestamp = datetime.now().strftime(TIMESTAMP_FORMAT)
        self.modelfile = MODEL_ROOT + self.tf + "/" + self.source_species + "_trained/basic_model/" + timestamp + "_run" + str(self.run)

        self.source_genome_file = GENOMES[self.source_species]
        self.target_genome_file = [GENOMES[target] for target in self.target_species]


    def get_output_path(self):
        return self.modelfile.split(".")[0] + ".probs.out"


    def set_steps_per_epoch(self):
        # NOTE: here we are assuming that the training set is balanced (50% bound examples)
        command = ["wc", "-l", self.bindingtrainposfile]
        linecount = int(check_output(command).strip().split()[0])
        self.train_steps = int(floor((linecount * 2) / self.batchsize))


    def set_chromsize(self):
        # leftover from when accessibility was also input to model
        command = ["head", "-n1", self.bindingtrainposfile]
        line1 = check_output(command).strip()
        # should be 5 columns besides chromatin info in file
        self.chromsize = len(line1.split()) - 5


    def set_val_labels(self):
        # to avoid doing this repeatedly later, we load in all binary labels for val set now
        self.target_val_labels = []
        self.target_val_steps = []
        self.target_val_labels = []
        for target in self.targetvalfile:
            with open(target) as f:
                labels = np.array([int(line.split()[-1]) for line in f])
            steps = int(floor(labels.shape[0] / self.valbatchsize))
            self.target_val_steps.append(steps)
            labels = labels[:steps * self.valbatchsize]
            self.target_val_labels.append(labels)

        with open(self.sourcevalfile) as f:
            self.source_val_labels = np.array([int(line.split()[-1]) for line in f])
        self.source_val_steps = int(floor(self.source_val_labels.shape[0] / self.valbatchsize))
        self.source_val_labels = self.source_val_labels[:self.source_val_steps * self.valbatchsize]

