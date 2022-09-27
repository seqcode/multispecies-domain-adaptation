from __future__ import division
from default_params import *
from math import ceil, floor
from subprocess import check_output
from pprint import pprint
import numpy as np


SPECIES_FILENAME = "chr3toY_shuf_runX_1E.bed"

# TRAIN_POS_FILENAME = "chr3toY_pos_shuf_small.bed"



class DA_Single_Params(Params):
    # This class is a subclass of the Params class in default_params.py.

    def __init__(self, args):
        Params.__init__(self, args)
        self.parse_args(args)

    def parse_args(self, args):
        assert len(args) >= 4, len(args)
        self.tf = args[1]
        assert self.tf in TFS, self.tf
        self.target_species = args[2]
        assert self.target_species in SPECIES, self.target_species
        self.run = int(args[3])
        self.train_species = args[4]
        assert self.train_species in SPECIES, self.target_species

        target_root = DATA_ROOT + self.target_species + "/" + self.tf + "/"
        
        self.source_species = self.train_species
        self.all_species = [self.train_species, self.target_species]
        
        source_root = DATA_ROOT + self.source_species + "/" + self.tf + "/"

        self.bindingtrainposfile = source_root + TRAIN_POS_FILENAME
        self.bindingtrainnegfile = source_root + TRAIN_NEG_FILENAME
        self.bindingtrainnegfile = self.bindingtrainnegfile.replace("runX", "run" + str(self.run))

        self.targetvalfile = target_root + VAL_FILENAME
        self.sourcevalfile = source_root + VAL_FILENAME

        timestamp = datetime.now().strftime(TIMESTAMP_FORMAT)
        self.modelfile = MODEL_ROOT + self.tf + "/" + self.train_species + "_trained/DA_single_model/" + timestamp + "_run" + str(self.run)

        self.target_genome_file = GENOMES[self.target_species]
        self.source_genome_file = GENOMES[self.source_species]

        self.target_species_file = target_root + SPECIES_FILENAME.replace("runX", "run" + str(self.run))
        self.source_species_file = source_root + SPECIES_FILENAME.replace("runX", "run" + str(self.run))
        self.lamb = 1
        self.loss_weight = 1


    def get_reshape_size(self):
        # DA models contain a reshape layer above the convolutional filters/pooling.
        # That layer needs, as input when initialized, what shape of input to expect.
        tmp = ceil(self.seqlen / self.strides)
        return int(tmp * self.convfilters)

    def set_val_labels(self):
        # to avoid doing this repeatedly later, we load in all binary labels for val set now
        self.target_val_labels = []
        self.target_val_steps = []
        self.target_val_labels = []
        with open(self.targetvalfile) as f:
            labels = np.array([int(line.split()[-1]) for line in f])
        steps = int(floor(labels.shape[0] / self.valbatchsize))
        self.target_val_steps.append(steps)
        labels = labels[:steps * self.valbatchsize]
        self.target_val_labels.append(labels)

        with open(self.sourcevalfile) as f:
            self.source_val_labels = np.array([int(line.split()[-1]) for line in f])
        self.source_val_steps = int(floor(self.source_val_labels.shape[0] / self.valbatchsize))
        self.source_val_labels = self.source_val_labels[:self.source_val_steps * self.valbatchsize]

