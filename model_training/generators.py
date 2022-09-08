import keras
import tensorflow
from keras.utils import Sequence
import numpy as np
import random
from math import ceil, floor

from seqdataloader.batchproducers.coordbased.core import Coordinates
from seqdataloader.batchproducers.coordbased.coordstovals.fasta import PyfaidxCoordsToVals

import os
import signal
import time
from itertools import islice



class TrainGenerator(Sequence):
    def __init__(self, params):
        self.posfile = params.bindingtrainposfile
        self.negfile = params.bindingtrainnegfile
        self.converter = PyfaidxCoordsToVals(params.source_genome_file)
        self.batchsize = params.batchsize
        self.halfbatchsize = self.batchsize // 2
        self.steps_per_epoch = params.train_steps
        self.total_epochs = params.epochs
        self.current_epoch = 1

        self.get_coords()
        self.on_epoch_end()


    def __len__(self):
        return self.steps_per_epoch


    def get_coords(self):
        # Using current filenames stored in self.posfile and self.negfile,
        # load in all of the training data as coordinates only.
        # Then, when it is time to fetch individual batches, a chunk of
        # coordinates will be converted into one-hot encoded sequences
        # ready for model input.
        try:
            with open(self.posfile) as posf:
                pos_coords_tmp = [line.split()[:3] for line in posf]  # expecting bed file format
                self.pos_coords = [Coordinates(coord[0], int(coord[1]), int(coord[2])) for coord in pos_coords_tmp]  # no strand consideration
            with open(self.negfile) as negf:
                neg_coords_tmp = [line.split()[:3] for line in negf]
                self.neg_coords = [Coordinates(coord[0], int(coord[1]), int(coord[2])) for coord in neg_coords_tmp]
        except Exception as e:  # this is here to circumvent strange code stalling issues on the cluster
            print(e)
            raise e


    def __getitem__(self, batch_index): 
        try:
            # First, get chunk of coordinates
            pos_coords_batch = self.pos_coords[batch_index * self.halfbatchsize : (batch_index + 1) * self.halfbatchsize]
            neg_coords_batch = self.neg_coords[batch_index * self.halfbatchsize : (batch_index + 1) * self.halfbatchsize]

            # if train_steps calculation is off, lists of coords may be empty
            assert len(pos_coords_batch) > 0, len(pos_coords_batch)
            assert len(neg_coords_batch) > 0, len(neg_coords_batch)

            # Seconds, convert the coordinates into one-hot encoded sequences
            pos_onehot = self.converter(pos_coords_batch)
            neg_onehot = self.converter(neg_coords_batch)

            # seqsdataloader returns empty array if coords are empty list or not in genome
            assert pos_onehot.shape[0] > 0, pos_onehot.shape[0]
            assert neg_onehot.shape[0] > 0, neg_onehot.shape[0]

            # Third, combine bound and unbound sites into one large array, and create label vector
            all_seqs = np.concatenate((pos_onehot, neg_onehot))
            labels = np.concatenate((np.ones(pos_onehot.shape[0],), np.zeros(neg_onehot.shape[0],)))

            assert all_seqs.shape[0] == self.batchsize, all_seqs.shape[0]
            return all_seqs, labels
        except Exception as e:  # this is here to circumvent strange code stalling issues on the cluster
            print(e)
            raise e


    def on_epoch_end(self):
        try:
            # switch to next set of negative examples
            prev_epoch = self.current_epoch
            next_epoch = prev_epoch + 1

            # update file where we will retrieve unbound site coordinates from
            prev_negfile = self.negfile
            next_negfile = prev_negfile.replace(str(prev_epoch) + "E", str(next_epoch) + "E")
            self.negfile = next_negfile

            # load in new unbound site coordinates
            with open(self.negfile) as negf:
                neg_coords_tmp = [line.split()[:3] for line in negf]
                self.neg_coords = [Coordinates(coord[0], int(coord[1]), int(coord[2])) for coord in neg_coords_tmp]
        
            # then shuffle positive examples
            random.shuffle(self.pos_coords)

        except Exception as e:  # this is here to circumvent strange code stalling issues on the cluster
            print(e)
            raise e

class MultiTrainGenerator(Sequence):
    def __init__(self, params):
        self.posfile = params.bindingtrainposfile
        self.negfile = params.bindingtrainnegfile
        self.converter = [PyfaidxCoordsToVals(genome_file) for genome_file in params.source_genome_file]
        self.batchsize = params.batchsize
        self.halfbatchsize = self.batchsize // 2
        self.steps_per_epoch = params.train_steps
        self.total_epochs = params.epochs
        self.current_epoch = 1
        self.seqlen = params.seqlen

        self.examples_needed = self.halfbatchsize*self.steps_per_epoch

        self.get_coords()
        self.on_epoch_end()


    def __len__(self):
        return self.steps_per_epoch


    def get_coords(self):
        try:
            # Using current filenames stored in self.posfile and self.negfile,
            # load in all of the "binding" training data as coordinates only.
            # Then, when it is time to fetch individual batches, a chunk of
            # coordinates will be converted into one-hot encoded sequences
            # ready for model input.
            self.pos_coords = []
            self.neg_coords = []

            for i in range(len(self.posfile)):
                with open(self.posfile[i]) as posf:
                    coords_tmp = [line.split()[:3] for line in posf]
                    coords_tmp = [Coordinates(coord[0], int(coord[1]), int(coord[2])) for coord in coords_tmp]
                    self.pos_coords += [[self.converter[i], coord] for coord in coords_tmp]
            random.shuffle(self.pos_coords)

            for i in range(len(self.negfile)):
                with open(self.negfile[i]) as negf:
                    lines_needed = list(islice(negf, self.examples_needed))
                    coords_tmp = [line.split()[:3] for line in lines_needed]
                    coords_tmp = [Coordinates(coord[0], int(coord[1]), int(coord[2])) for coord in coords_tmp]
                    self.neg_coords += [[self.converter[i], coord] for coord in coords_tmp]
            random.shuffle(self.neg_coords)

        except Exception as e:
            print(e)
            raise e


    def __getitem__(self, batch_index):
        try:

            
            # First, we retrieve a chunk of coordinates for both the bound and unbound site examples,
            # and convert those coordinates to one-hot encoded sequence arrays

            pos_onehot = np.empty(shape=(self.halfbatchsize, self.seqlen, 4), dtype=int)
            neg_onehot = np.empty(shape=(self.halfbatchsize, self.seqlen, 4), dtype=int)


            for i in range(batch_index * self.halfbatchsize, (batch_index + 1) * self.halfbatchsize):
                pos_onehot[i%self.halfbatchsize] = self.pos_coords[i][0]([self.pos_coords[i][1]])
                neg_onehot[i%self.halfbatchsize] = self.neg_coords[i][0]([self.neg_coords[i][1]])

            assert pos_onehot.shape[0] > 0, pos_onehot.shape[0]
            assert neg_onehot.shape[0] > 0, neg_onehot.shape[0]


            all_seqs = np.concatenate((pos_onehot, neg_onehot))

            # label vector for binding prediction task:
            labels = np.concatenate((np.ones(pos_onehot.shape[0],), np.zeros(neg_onehot.shape[0],)))
            
            return all_seqs, labels
        except Exception as e:
            print(e)
            raise e


    def on_epoch_end(self):
        try:
            # switch to next set of negative and species examples
            prev_epoch = self.current_epoch
            next_epoch = prev_epoch + 1

            # update file to pull coordinates from, for unbound examples and species-background examples
            prev_negfile = self.negfile
            next_negfile = [negfile.replace(str(prev_epoch) + "E", str(next_epoch) + "E") for negfile in prev_negfile]
            self.negfile = next_negfile

            # load in coordinates into memory for unbound examples
            
            for i in range(len(self.negfile)):
                with open(self.negfile[i]) as negf:
                    lines_needed = list(islice(negf, self.examples_needed))
                    coords_tmp = [line.split()[:3] for line in lines_needed]
                    coords_tmp = [Coordinates(coord[0], int(coord[1]), int(coord[2])) for coord in coords_tmp]
                    self.neg_coords += [[self.converter[i], coord] for coord in coords_tmp]

            # then shuffle
            random.shuffle(self.neg_coords)
            random.shuffle(self.pos_coords)

        except Exception as e:
            print(e)
            raise e


class EnsembleTrainGenerator(Sequence):
    def __init__(self, params):
        self.posfile = params.bindingtrainposfile
        self.negfile = params.bindingtrainnegfile
        self.converter = PyfaidxCoordsToVals(params.target_genome_file)
        self.batchsize = params.batchsize
        self.halfbatchsize = self.batchsize // 2
        self.steps_per_epoch = params.train_steps
        self.total_epochs = params.epochs
        self.current_epoch = 1
        self.source_species = params.source_species
        self.tf = params.tf
        self.run = params.run

        self.examples_needed = self.halfbatchsize*self.steps_per_epoch

        self.get_coords()
        self.on_epoch_end()

        


    def __len__(self):
        return self.steps_per_epoch


    def get_coords(self):
        # Using current filenames stored in self.posfile and self.negfile,
        # load in all of the training data as coordinates only.
        # Then, when it is time to fetch individual batches, a chunk of
        # coordinates will be converted into one-hot encoded sequences
        # ready for model input.
        try:
            with open(self.posfile) as posf:
                pos_coords_tmp = [line.split()[:3] for line in posf]  # expecting bed file format
                self.pos_coords = [Coordinates(coord[0], int(coord[1]), int(coord[2])) for coord in pos_coords_tmp]  # no strand consideration
            with open(self.negfile) as negf:
                lines_needed = list(islice(negf, self.examples_needed))
                neg_coords_tmp = [line.split()[:3] for line in lines_needed]
                self.neg_coords = [Coordinates(coord[0], int(coord[1]), int(coord[2])) for coord in neg_coords_tmp]
        except Exception as e:  # this is here to circumvent strange code stalling issues on the cluster
            print(e)
            raise e

    def get_model_file(tf, train_species, run = 1, model_type = "basic_model"):
        # This function returns the filepath where the model for a given
        # TF, training species, and run is saved.
        # By default, the file for the best model across all training epochs
        # is returned, you can change model_type to select the last model instead.
        # This function specifically looks for the most recent model file,
        # if there are multiple for the same run-TF-species combo.
        try:
            run_int = int(run)
        except:
            print("Error: You need to pass in a run number that can be cast to int.")
        
        model_file_prefix = ROOT + "/".join(["models", tf, train_species + "_trained", model_type]) + "/"      
        
        # these models were saved as part of training
        # see ../2_train_and_test_models/callbacks.py for model saving details 
        try:
            model_file_suffix = "_run" + str(run) + "_best.model"
        except:
            model_file_suffix = "_run" + str(run) + "_15E_end.model"
        
        # get all files that match the prefix and suffix
        files = [f for f in os.listdir(model_file_prefix) if f.endswith(model_file_suffix)]
        
        # sort files and return the one that is most recent
        latest_file = max([model_file_prefix + f for f in files], key=os.path.getctime)
        return latest_file


    def __getitem__(self, batch_index): 
        try:
            # First, get chunk of coordinates
            pos_coords_batch = self.pos_coords[batch_index * self.halfbatchsize : (batch_index + 1) * self.halfbatchsize]
            neg_coords_batch = self.neg_coords[batch_index * self.halfbatchsize : (batch_index + 1) * self.halfbatchsize]

            # if train_steps calculation is off, lists of coords may be empty
            assert len(pos_coords_batch) > 0, len(pos_coords_batch)
            assert len(neg_coords_batch) > 0, len(neg_coords_batch)

            # Seconds, convert the coordinates into one-hot encoded sequences
            pos_onehot = self.converter(pos_coords_batch)
            neg_onehot = self.converter(neg_coords_batch)

            # seqsdataloader returns empty array if coords are empty list or not in genome
            assert pos_onehot.shape[0] > 0, pos_onehot.shape[0]
            assert neg_onehot.shape[0] > 0, neg_onehot.shape[0]

            # Third, combine bound and unbound sites into one large array, and create label vector
            all_seqs = np.concatenate((pos_onehot, neg_onehot))

            # all_preds = np.empty(shape=(all_seqs.shape[0], 0))

            # for source in self.source_species:
            #     model_file = get_model_file(self.tf, source, self.run)
            #     mdl = keras.models.load_model(model_file)
            #     preds = mdl.predict(all_seqs, 1024, use_multiprocessing = True, workers = 8, verbose = 1)
            #     all_preds = np.concatenate((all_preds, preds), axis=1)

            labels = np.concatenate((np.ones(pos_onehot.shape[0],), np.zeros(neg_onehot.shape[0],)))

            # assert all_seqs.shape[0] == self.batchsize, all_seqs.shape[0]
            return all_seqs, labels
        except Exception as e:  # this is here to circumvent strange code stalling issues on the cluster
            print(e)
            raise e


    def on_epoch_end(self):
        try:
            # switch to next set of negative examples
            prev_epoch = self.current_epoch
            next_epoch = prev_epoch + 1

            # update file where we will retrieve unbound site coordinates from
            prev_negfile = self.negfile
            next_negfile = prev_negfile.replace(str(prev_epoch) + "E", str(next_epoch) + "E")
            self.negfile = next_negfile

            # load in new unbound site coordinates
            with open(self.negfile) as negf:
                lines_needed = list(islice(negf, self.examples_needed))
                neg_coords_tmp = [line.split()[:3] for line in lines_needed]
                self.neg_coords = [Coordinates(coord[0], int(coord[1]), int(coord[2])) for coord in neg_coords_tmp]
        
            # then shuffle positive examples
            random.shuffle(self.pos_coords)

        except Exception as e:  # this is here to circumvent strange code stalling issues on the cluster
            print(e)
            raise e


class ValGenerator(Sequence):
    def __init__(self, params, target_species = None, source_species = None):
        if target_species:
            if isinstance(params.target_species, list):
                ind = params.target_species.index(target_species)
                self.valfile = params.targetvalfile[ind]
                self.steps_per_epoch = params.target_val_steps[ind]
                self.converter = PyfaidxCoordsToVals(params.target_genome_file[ind])
            else:
                self.valfile = params.targetvalfile
                self.steps_per_epoch = params.target_val_steps
                self.converter = PyfaidxCoordsToVals(params.target_genome_file)
        elif source_species:
            if isinstance(params.source_species, list):
                ind = params.source_species.index(source_species)
                self.valfile = params.sourcevalfile[ind]
                self.steps_per_epoch = params.source_val_steps[ind]
                self.converter = PyfaidxCoordsToVals(params.source_genome_file[ind])
            else:
                self.valfile = params.sourcevalfile
                self.steps_per_epoch = params.source_val_steps
                self.converter = PyfaidxCoordsToVals(params.source_genome_file)
            
        self.batchsize = params.valbatchsize
        self.get_coords()


    def __len__(self):
        return self.steps_per_epoch


    def get_coords(self):
        # load in coordinates of each validation set example into memory

        # start = time.time()
        try:
            with open(self.valfile) as f:
                coords_tmp = [line.split()[:3] for line in f]
                self.coords = [Coordinates(coord[0], int(coord[1]), int(coord[2])) for coord in coords_tmp]
        except Exception as e:  # this is here to circumvent strange code stalling issues on the cluster
            print(e)
            raise e

        # end = time.time()
        # t = end-start
        # with open('DA_time_generator.txt', "a") as f:
        #     f.write(f"Val get coords took {t} time, ended at {end}\n")


    def __getitem__(self, batch_index):

        # start = time.time()
        try:
            # get chunk of coordinates
            coords_batch = self.coords[batch_index * self.batchsize : (batch_index + 1) * self.batchsize]
            assert len(coords_batch) > 0, len(coords_batch)

            # convert chunk of coordinates to array of one-hot encoded sequences
            seq_onehot = self.converter(coords_batch)
            assert seq_onehot.shape[0] > 0, seq_onehot.shape[0] 
            return seq_onehot
        except Exception as e:  # this is here to circumvent strange code stalling issues on the cluster
            print(e)
            raise e

        # end = time.time()
        # t = end-start
        # with open('DA_time_generator.txt', "a") as f:
        #     f.write(f"Val get item took {t} time, ended at {end}\n")



class DATrainGenerator(Sequence):
    def __init__(self, params):
        print(vars(params))

        self.posfile = params.bindingtrainposfile
        self.negfile = params.bindingtrainnegfile
        self.source_species_file = params.source_species_file
        self.target_species_file = params.target_species_file
        self.target_converter = PyfaidxCoordsToVals(params.target_genome_file)
        self.source_converter = [PyfaidxCoordsToVals(genome_file) for genome_file in params.source_genome_file]
        self.batchsize = params.batchsize
        self.halfbatchsize = self.batchsize // 2
        self.steps_per_epoch = params.train_steps
        self.total_epochs = params.epochs
        self.current_epoch = 1

        self.all_species = params.all_species
        self.num_species = len(self.all_species)
        self.target_species = params.target_species
        self.seqlen = params.seqlen

        self.examples_needed = self.halfbatchsize*self.steps_per_epoch


        # start = time.time()
        self.get_binding_coords()
        self.get_species_coords()
        # end = time.time()
        # t = end-start
        # with open('DA_time_generator.txt', "a") as f:
        #     f.write(f"Get coords took {t} time, ended at {end}\n")
        self.on_epoch_end()


    def __len__(self):
        return self.steps_per_epoch


    def get_binding_coords(self):
        try:
            # Using current filenames stored in self.posfile and self.negfile,
            # load in all of the "binding" training data as coordinates only.
            # Then, when it is time to fetch individual batches, a chunk of
            # coordinates will be converted into one-hot encoded sequences
            # ready for model input.
            self.pos_coords = []
            self.neg_coords = []
            for i in range(len(self.posfile)):
                with open(self.posfile[i]) as posf:
                    coords_tmp = [line.split()[:3] for line in posf]
                    coords_tmp = [Coordinates(coord[0], int(coord[1]), int(coord[2])) for coord in coords_tmp]
                    self.pos_coords += [[self.source_converter[i], coord] for coord in coords_tmp]

            random.shuffle(self.pos_coords)
            
            for i in range(len(self.negfile)):
                with open(self.negfile[i]) as negf:
                    lines_needed = list(islice(negf, self.examples_needed))
                    coords_tmp = [line.split()[:3] for line in lines_needed]
                    coords_tmp = [Coordinates(coord[0], int(coord[1]), int(coord[2])) for coord in coords_tmp]
                    self.neg_coords += [[self.source_converter[i], coord] for coord in coords_tmp]

            random.shuffle(self.neg_coords)

        except Exception as e:
            print(e)
            raise e


    def get_species_coords(self):
        try:
            self.source_coords = []
            # Same as get_binding_coords(), but loading in coordinates
            # for DA-specific training data (from both species).
            with open(self.target_species_file) as targetf:
                lines_needed = list(islice(targetf, self.examples_needed))
                target_coords_tmp = [line.split()[:3] for line in lines_needed]
                self.target_coords = [Coordinates(coord[0], int(coord[1]), int(coord[2])) for coord in target_coords_tmp]
            
            for i in range(len(self.source_species_file)):
                with open(self.source_species_file[i]) as sourcef:
                    lines_needed = list(islice(sourcef, self.examples_needed))
                    coords_tmp = [line.split()[:3] for line in lines_needed]
                    coords_tmp = [Coordinates(coord[0], int(coord[1]), int(coord[2])) for coord in coords_tmp]
                    self.source_coords += [[i, self.source_converter[i], coord] for coord in coords_tmp]

            random.shuffle(self.source_coords)


            # for source in self.source_species_file:
            #     source_tmp = []
            #     with open(source) as sourcef:
            #         source_coords_tmp = [line.split()[:3] for line in sourcef]
            #         coords = [Coordinates(coord[0], int(coord[1]), int(coord[2])) for coord in source_coords_tmp]
            #         source_tmp.append(coords)
            #     self.source_coords.append(source_tmp)
        except Exception as e:
            print(e)
            raise e


    def __getitem__(self, batch_index):
        try:

            
            # First, we retrieve a chunk of coordinates for both the bound and unbound site examples,
            # and convert those coordinates to one-hot encoded sequence arrays

            # start = time.time()

            target_per_step = min(int(floor(len(self.target_coords) / self.steps_per_epoch)), self.halfbatchsize)

            pos_onehot = np.empty(shape=(self.halfbatchsize, self.seqlen, 4), dtype=int)
            neg_onehot = np.empty(shape=(self.halfbatchsize, self.seqlen, 4), dtype=int)
            source_onehot = np.empty(shape=(self.halfbatchsize, self.seqlen, 4), dtype=int)
            target_onehot = np.empty(shape=(target_per_step, self.seqlen, 4), dtype=int)


            source_indices = []
            for i in range(batch_index * self.halfbatchsize, (batch_index + 1) * self.halfbatchsize):
                pos_onehot[i%self.halfbatchsize] = self.pos_coords[i][0]([self.pos_coords[i][1]])
                neg_onehot[i%self.halfbatchsize] = self.neg_coords[i][0]([self.neg_coords[i][1]])
                source_onehot[i%self.halfbatchsize] = self.source_coords[i][1]([self.source_coords[i][2]])
                source_indices.append(self.source_coords[i][0])


            for i in range(batch_index * target_per_step, (batch_index + 1) * target_per_step):
                target_onehot[i%target_per_step] = self.target_converter([self.target_coords[i]])

            assert pos_onehot.shape[0] > 0, pos_onehot.shape[0]
            assert neg_onehot.shape[0] > 0, neg_onehot.shape[0]


            # Second, we do the same thing again, but for the "species-background" data of both species

            assert source_onehot.shape[0] > 0, source_onehot.shape[0]

            assert target_onehot.shape[0] > 0, target_onehot.shape[0]

            
            # Third, concatenate all the one-hot encoded sequences together


            all_seqs = np.concatenate((pos_onehot, neg_onehot, source_onehot, target_onehot))

            # Fourth, create label vectors for both tasks
            # Note that a label of -1 will correspond to a masking of the loss function
            # (so if the label for the binding task is -1 for example i, then when the
            # loss gradient backpropagates, example i will not be included in that calculation

            # label vector for binding prediction task:
            binding_labels = np.concatenate((np.ones(pos_onehot.shape[0],), np.zeros(neg_onehot.shape[0],),
                                        -1 * np.ones(source_onehot.shape[0],), -1 * np.ones(target_onehot.shape[0],)))

            # end = time.time()
            # t = end-start
            # with open('DA_time_generator.txt', "a") as f:
            #     f.write(f"Making binding labels took {t} time, ended at {end}\n")
            
            # label vector for species discrimination task:

            # start = time.time()

            source_species = self.all_species[:]
            source_species.remove(self.target_species)

            source_labels, target_labels = [], []

            for source_index in source_indices:
                ind = self.all_species.index(source_species[source_index])
                onehot = [0]*self.num_species
                onehot[ind] = 1
                source_labels.append(onehot)

            ind = self.all_species.index(self.target_species)
            onehot = [0]*self.num_species
            onehot[ind] = 1
            target_labels = [onehot]*target_onehot.shape[0]

            source_labels = np.array(source_labels)
            target_labels = np.array(target_labels)


            species_labels = np.concatenate((-1 * np.ones((pos_onehot.shape[0], self.num_species)), -1 * np.ones((neg_onehot.shape[0], self.num_species)),
                                        source_labels, target_labels))

            # end = time.time()
            # t = end-start
            # with open('DA_time_generator.txt', "a") as f:
            #     f.write(f"Making species labels took {t} time, ended at {end}\n")

            # assert all_seqs.shape[0] == self.batchsize * 2, all_seqs.shape[0]
            # assert binding_labels.shape == species_labels.shape, (binding_labels.shape, species_labels.shape)

            # here we assign the name "discriminator" to the binding prediction task, and
            # "classifier" to the species prediction task
            return all_seqs, {"discriminator":binding_labels, "classifier":species_labels}
        except Exception as e:
            print(e)
            raise e


    def on_epoch_end(self):
        try:
            # switch to next set of negative and species examples
            prev_epoch = self.current_epoch
            next_epoch = prev_epoch + 1

            # update file to pull coordinates from, for unbound examples and species-background examples
            prev_negfile = self.negfile
            next_negfile = [negfile.replace(str(prev_epoch) + "E", str(next_epoch) + "E") for negfile in prev_negfile]
            self.negfile = next_negfile

            prev_targetfile = self.target_species_file
            next_targetfile = prev_targetfile.replace(str(prev_epoch) + "E", str(next_epoch) + "E")
            self.target_species_file = next_targetfile

            prev_sourcefile = self.source_species_file
            next_sourcefile = [sourcefile.replace(str(prev_epoch) + "E", str(next_epoch) + "E") for sourcefile in prev_sourcefile]
            self.source_species_file = next_sourcefile


            # load in coordinates into memory for unbound examples and species-background examples

            # start = time.time()
            
            for i in range(len(self.negfile)):
                with open(self.negfile[i]) as negf:
                    lines_needed = list(islice(negf, self.examples_needed))
                    coords_tmp = [line.split()[:3] for line in lines_needed]
                    coords_tmp = [Coordinates(coord[0], int(coord[1]), int(coord[2])) for coord in coords_tmp]
                    self.neg_coords += [[self.source_converter[i], coord] for coord in coords_tmp]


            with open(self.target_species_file) as targetf:
                lines_needed = list(islice(targetf, self.examples_needed))
                target_coords_tmp = [line.split()[:3] for line in lines_needed]
                self.target_coords = [Coordinates(coord[0], int(coord[1]), int(coord[2])) for coord in target_coords_tmp]
            

            for i in range(len(self.source_species_file)):
                with open(self.source_species_file[i]) as sourcef:
                    lines_needed = list(islice(sourcef, self.examples_needed))
                    coords_tmp = [line.split()[:3] for line in lines_needed]
                    coords_tmp = [Coordinates(coord[0], int(coord[1]), int(coord[2])) for coord in coords_tmp]
                    self.source_coords += [[i, self.source_converter[i], coord] for coord in coords_tmp]

            random.shuffle(self.neg_coords)
            random.shuffle(self.source_coords)

            # then shuffle positive examples

            random.shuffle(self.pos_coords)

            # end = time.time()
            # t = end-start
            # with open('DA_time_generator.txt', "a") as f:
            #     f.write(f"Generator on epoch end took {t} time, ended at {end}\n")

        except Exception as e:
            print(e)
            raise e

class BATrainGenerator(Sequence):
    def __init__(self, params):
        print(vars(params))

        self.posfile = params.bindingtrainposfile
        self.negfile = params.bindingtrainnegfile
        self.source_species_file = params.source_species_file
        self.target_species_file = params.target_species_file
        self.target_converter = PyfaidxCoordsToVals(params.target_genome_file)
        self.source_converter = [PyfaidxCoordsToVals(genome_file) for genome_file in params.source_genome_file]
        self.batchsize = params.batchsize
        self.halfbatchsize = self.batchsize // 2
        self.steps_per_epoch = params.train_steps
        self.total_epochs = params.epochs
        self.current_epoch = 1

        self.all_species = params.all_species
        self.num_species = len(self.all_species)
        self.target_species = params.target_species
        self.seqlen = params.seqlen

        self.examples_needed = self.halfbatchsize*self.steps_per_epoch


        # start = time.time()
        self.get_binding_coords()
        self.get_species_coords()
        # end = time.time()
        # t = end-start
        # with open('DA_time_generator.txt', "a") as f:
        #     f.write(f"Get coords took {t} time, ended at {end}\n")
        self.on_epoch_end()


    def __len__(self):
        return self.steps_per_epoch


    def get_binding_coords(self):
        try:
            # Using current filenames stored in self.posfile and self.negfile,
            # load in all of the "binding" training data as coordinates only.
            # Then, when it is time to fetch individual batches, a chunk of
            # coordinates will be converted into one-hot encoded sequences
            # ready for model input.
            self.pos_coords = []
            self.neg_coords = []
            for i in range(len(self.posfile)):
                with open(self.posfile[i]) as posf:
                    coords_tmp = [line.split()[:3] for line in posf]
                    coords_tmp = [Coordinates(coord[0], int(coord[1]), int(coord[2])) for coord in coords_tmp]
                    self.pos_coords += [[self.source_converter[i], coord] for coord in coords_tmp]

            random.shuffle(self.pos_coords)
            
            for i in range(len(self.negfile)):
                with open(self.negfile[i]) as negf:
                    lines_needed = list(islice(negf, self.examples_needed))
                    coords_tmp = [line.split()[:3] for line in lines_needed]
                    coords_tmp = [Coordinates(coord[0], int(coord[1]), int(coord[2])) for coord in coords_tmp]
                    self.neg_coords += [[self.source_converter[i], coord] for coord in coords_tmp]

            random.shuffle(self.neg_coords)

        except Exception as e:
            print(e)
            raise e


    def get_species_coords(self):
        try:
            self.source_coords = []
            # Same as get_binding_coords(), but loading in coordinates
            # for DA-specific training data (from both species).
            with open(self.target_species_file) as targetf:
                lines_needed = list(islice(targetf, self.examples_needed))
                target_coords_tmp = [line.split()[:3] for line in lines_needed]
                self.target_coords = [Coordinates(coord[0], int(coord[1]), int(coord[2])) for coord in target_coords_tmp]
            
            for i in range(len(self.source_species_file)):
                with open(self.source_species_file[i]) as sourcef:
                    lines_needed = list(islice(sourcef, self.examples_needed))
                    coords_tmp = [line.split()[:3] for line in lines_needed]
                    coords_tmp = [Coordinates(coord[0], int(coord[1]), int(coord[2])) for coord in coords_tmp]
                    self.source_coords += [[i, self.source_converter[i], coord] for coord in coords_tmp]

            random.shuffle(self.source_coords)


            # for source in self.source_species_file:
            #     source_tmp = []
            #     with open(source) as sourcef:
            #         source_coords_tmp = [line.split()[:3] for line in sourcef]
            #         coords = [Coordinates(coord[0], int(coord[1]), int(coord[2])) for coord in source_coords_tmp]
            #         source_tmp.append(coords)
            #     self.source_coords.append(source_tmp)
        except Exception as e:
            print(e)
            raise e


    def __getitem__(self, batch_index):
        try:

            
            # First, we retrieve a chunk of coordinates for both the bound and unbound site examples,
            # and convert those coordinates to one-hot encoded sequence arrays

            # start = time.time()

            target_per_step = min(int(floor(len(self.target_coords) / self.steps_per_epoch)), self.halfbatchsize)

            pos_onehot = np.empty(shape=(self.halfbatchsize, self.seqlen, 4), dtype=int)
            neg_onehot = np.empty(shape=(self.halfbatchsize, self.seqlen, 4), dtype=int)
            source_onehot = np.empty(shape=(self.halfbatchsize, self.seqlen, 4), dtype=int)
            target_onehot = np.empty(shape=(target_per_step, self.seqlen, 4), dtype=int)


            source_indices = []
            for i in range(batch_index * self.halfbatchsize, (batch_index + 1) * self.halfbatchsize):
                pos_onehot[i%self.halfbatchsize] = self.pos_coords[i][0]([self.pos_coords[i][1]])
                neg_onehot[i%self.halfbatchsize] = self.neg_coords[i][0]([self.neg_coords[i][1]])
                source_onehot[i%self.halfbatchsize] = self.source_coords[i][1]([self.source_coords[i][2]])
                source_indices.append(self.source_coords[i][0])


            for i in range(batch_index * target_per_step, (batch_index + 1) * target_per_step):
                target_onehot[i%target_per_step] = self.target_converter([self.target_coords[i]])

            assert pos_onehot.shape[0] > 0, pos_onehot.shape[0]
            assert neg_onehot.shape[0] > 0, neg_onehot.shape[0]


            # Second, we do the same thing again, but for the "species-background" data of both species

            assert source_onehot.shape[0] > 0, source_onehot.shape[0]

            assert target_onehot.shape[0] > 0, target_onehot.shape[0]

            
            # Third, concatenate all the one-hot encoded sequences together


            all_seqs = np.concatenate((pos_onehot, neg_onehot, source_onehot, target_onehot))

            # Fourth, create label vectors for both tasks
            # Note that a label of -1 will correspond to a masking of the loss function
            # (so if the label for the binding task is -1 for example i, then when the
            # loss gradient backpropagates, example i will not be included in that calculation

            # label vector for binding prediction task:
            binding_labels = np.concatenate((np.ones(pos_onehot.shape[0],), np.zeros(neg_onehot.shape[0],),
                                        -1 * np.ones(source_onehot.shape[0],), -1 * np.ones(target_onehot.shape[0],)))

            species_labels = np.concatenate((-1 * np.ones(pos_onehot.shape[0],), -1 * np.ones(neg_onehot.shape[0],),
                                        np.zeros(source_onehot.shape[0],), np.ones(target_onehot.shape[0],)))


            return all_seqs, {"discriminator":binding_labels, "classifier":species_labels}
        except Exception as e:
            print(e)
            raise e


    def on_epoch_end(self):
        try:
            # switch to next set of negative and species examples
            prev_epoch = self.current_epoch
            next_epoch = prev_epoch + 1

            # update file to pull coordinates from, for unbound examples and species-background examples
            prev_negfile = self.negfile
            next_negfile = [negfile.replace(str(prev_epoch) + "E", str(next_epoch) + "E") for negfile in prev_negfile]
            self.negfile = next_negfile

            prev_targetfile = self.target_species_file
            next_targetfile = prev_targetfile.replace(str(prev_epoch) + "E", str(next_epoch) + "E")
            self.target_species_file = next_targetfile

            prev_sourcefile = self.source_species_file
            next_sourcefile = [sourcefile.replace(str(prev_epoch) + "E", str(next_epoch) + "E") for sourcefile in prev_sourcefile]
            self.source_species_file = next_sourcefile


            # load in coordinates into memory for unbound examples and species-background examples

            # start = time.time()
            
            for i in range(len(self.negfile)):
                with open(self.negfile[i]) as negf:
                    lines_needed = list(islice(negf, self.examples_needed))
                    coords_tmp = [line.split()[:3] for line in lines_needed]
                    coords_tmp = [Coordinates(coord[0], int(coord[1]), int(coord[2])) for coord in coords_tmp]
                    self.neg_coords += [[self.source_converter[i], coord] for coord in coords_tmp]


            with open(self.target_species_file) as targetf:
                lines_needed = list(islice(targetf, self.examples_needed))
                target_coords_tmp = [line.split()[:3] for line in lines_needed]
                self.target_coords = [Coordinates(coord[0], int(coord[1]), int(coord[2])) for coord in target_coords_tmp]
            

            for i in range(len(self.source_species_file)):
                with open(self.source_species_file[i]) as sourcef:
                    lines_needed = list(islice(sourcef, self.examples_needed))
                    coords_tmp = [line.split()[:3] for line in lines_needed]
                    coords_tmp = [Coordinates(coord[0], int(coord[1]), int(coord[2])) for coord in coords_tmp]
                    self.source_coords += [[i, self.source_converter[i], coord] for coord in coords_tmp]

            random.shuffle(self.neg_coords)
            random.shuffle(self.source_coords)

            # then shuffle positive examples

            random.shuffle(self.pos_coords)

            # end = time.time()
            # t = end-start
            # with open('DA_time_generator.txt', "a") as f:
            #     f.write(f"Generator on epoch end took {t} time, ended at {end}\n")

        except Exception as e:
            print(e)
            raise e


