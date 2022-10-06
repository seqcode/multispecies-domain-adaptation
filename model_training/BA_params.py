from __future__ import division
from default_params import *
from math import ceil, floor
from subprocess import check_output
from pprint import pprint
import numpy as np


SPECIES_FILENAME = "chr3toY_shuf_runX_1E.bed"

TRAIN_POS_FILENAME = "chr3toY_pos_shuf_small.bed"



class BA_Params(Params):
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

        target_root = DATA_ROOT + self.target_species + "/" + self.tf + "/"
        if(self.tf == 'CEBPA'):
            self.source_species = [species for species in SPECIES if species != self.target_species]
            self.all_species = SPECIES
        else:
            self.source_species = [species for species in SPECIES_SMALL if species != self.target_species]
            self.all_species = SPECIES_SMALL
        
        source_root = [DATA_ROOT + source + "/" + self.tf + "/" for source in self.source_species]

        self.bindingtrainposfile = [root + TRAIN_POS_FILENAME for root in source_root]
        self.bindingtrainnegfile = [root + TRAIN_NEG_FILENAME for root in source_root]
        self.bindingtrainnegfile = [negfile.replace("runX", "run" + str(self.run)) for negfile in self.bindingtrainnegfile]

        self.targetvalfile = target_root + VAL_FILENAME
        self.sourcevalfile = [source + VAL_FILENAME for source in source_root]

        timestamp = datetime.now().strftime(TIMESTAMP_FORMAT)
        self.modelfile = MODEL_ROOT + self.tf + "/" + self.target_species + "_trained/BA_model/" + timestamp + "_run" + str(self.run)

        self.target_genome_file = GENOMES[self.target_species]
        self.source_genome_file = [GENOMES[source] for source in self.source_species]

        self.target_species_file = target_root + SPECIES_FILENAME.replace("runX", "run" + str(self.run))
        self.source_species_file = [source + SPECIES_FILENAME.replace("runX", "run" + str(self.run)) for source in source_root]
        self.lamb = 1
        self.loss_weight = 1


    def get_reshape_size(self):
        # DA models contain a reshape layer above the convolutional filters/pooling.
        # That layer needs, as input when initialized, what shape of input to expect.
        tmp = ceil(self.seqlen / self.strides)
        return int(tmp * self.convfilters)


    def set_steps_per_epoch(self):
        total_linecount = 0
        for posfile in self.bindingtrainposfile:
            command = ["wc", "-l", posfile]
            linecount = int(check_output(command).strip().split()[0])
            total_linecount += linecount
        print("Total linecount:" , total_linecount)
        self.train_steps = int(floor(total_linecount / (self.batchsize / 2))) ###
        print("Train steps:", self.train_steps)


    def set_val_labels(self):
        # to avoid doing this repeatedly later, we load in all binary labels for val set now
        self.source_val_labels = []
        self.source_val_steps = []
        self.source_val_labels = []
        for source in self.sourcevalfile:
            with open(source) as f:
                labels = np.array([int(line.split()[-1]) for line in f])
            steps = int(floor(labels.shape[0] / self.valbatchsize))
            self.source_val_steps.append(steps)
            labels = labels[:steps * self.valbatchsize]
            self.source_val_labels.append(labels)

        with open(self.targetvalfile) as f:
            self.target_val_labels = np.array([int(line.split()[-1]) for line in f])
        self.target_val_steps = int(floor(self.target_val_labels.shape[0] / self.valbatchsize))
        self.target_val_labels = self.target_val_labels[:self.target_val_steps * self.valbatchsize]

