import os, sys
import keras
import numpy as np
from collections import defaultdict
import joblib


from keras.utils import Sequence
from seqdataloader.batchproducers.coordbased.core import Coordinates
from seqdataloader.batchproducers.coordbased.coordstovals.fasta import PyfaidxCoordsToVals

from flipGradientTF import GradientReversal
import tensorflow

from keras.losses import binary_crossentropy, categorical_crossentropy


ROOT = "/storage/home/vza5092/group/projects/cross-species/liver-tfs/"

GENOMES = {"mm10" : "/storage/home/vza5092/group/genomes/mm10/mm10.fa",
       "hg38" : "/storage/home/vza5092/group/genomes/hg38/hg38.fa",
       "monDom5" : "/storage/home/vza5092/group/genomes/monDom5/monDom5.fa",
       "canFam4" : "/storage/home/vza5092/group/genomes/canFam4/canFam4.fa",
       "galGal6" : "/storage/home/vza5092/group/genomes/galGal6/galGal6.fa",
       "rn5" : "/storage/home/vza5092/group/genomes/rn5/rn5.fa",
       "rheMac10" : "/storage/home/vza5092/group/genomes/rheMac10/rheMac10.fa"}

SPECIES = ["mm10", "hg38", "monDom5", "canFam4", "galGal6", "rn5", "rheMac10"]
SPECIES_SMALL = ["mm10", "hg38", "canFam4", "rn5", "rheMac10"]

model_types = ["kelly", "basic_model"]

def get_test_bed_file(species):
    # This function returns the path to a BED-format file
    # containing the chromosome names, starts, and ends for
    # all examples to test the model with.
    # Note binding labels will not be loaded in.
    # This file should contain the same examples for any TF.
    return(ROOT + "exp_data/" + species + "/CEBPA/chr2.bed")


class ValGenerator(Sequence):
    # This generator retrieves all coordinates for windows in the test set
    # and converts the sequences in those windows to one-hot encodings.
    # Which species to retrieve test windows for is specified with
    # the "val_species" argument. 
    
    def __init__(self, batchsize, val_species = "hg38"):
        self.valfile = get_test_bed_file(val_species)
        self.get_steps(batchsize)
        self.converter = PyfaidxCoordsToVals(GENOMES[val_species])
        self.batchsize = batchsize
        self.get_coords()
        
        
    def get_steps(self, batchsize):
        # calculates the number of steps needed to get through
        # all batches of examples in the test dataset
        # (Keras predict_generator code needs to know this)
        with open(self.valfile) as f:
            lines_in_file = sum(1 for line in f)
        
        self.steps = lines_in_file // batchsize


    def __len__(self):
        return self.steps

    def get_coords(self):
        # load all coordinates for the test data into memory
        with open(self.valfile) as f:
            coords_tmp = [line.rstrip().split()[:3] for line in f]
            
        assert [len(line_split) == 3 for line_split in coords_tmp]
        self.coords = [Coordinates(coord[0], int(coord[1]), int(coord[2])) for coord in coords_tmp]

    def __getitem__(self, batch_index):
        # convert a batch's worth of coordinates into one-hot sequences
        batch = self.coords[batch_index * self.batchsize : (batch_index + 1) * self.batchsize]
        return self.converter(batch)
    

def get_preds_batched_fast(model, batch_size, test_species = "hg38"):
    # Make predictions for all test data using a specified model.
    # Batch_size can be as big as your compute can handle.
    # Use test_species = "mm10" to test on mouse data instead of human data.
    
    print("Generating predictions...")
    current_probs = model.predict_generator(ValGenerator(batch_size, test_species),
                                               use_multiprocessing = True, workers = 8, verbose = 1)
    if len(current_probs) == 2:
        current_probs = current_probs[0]
    return np.squeeze(current_probs)



def custom_loss_disc(y_true, y_pred):
    # The model will be trained using this loss function, which is identical to normal BCE
    # except when the label for an example is -1, that example is masked out for that task.
    # This allows for examples to only impact loss backpropagation for one of the two tasks.
    y_pred = tensorflow.boolean_mask(y_pred, tensorflow.not_equal(y_true, -1))
    y_true = tensorflow.boolean_mask(y_true, tensorflow.not_equal(y_true, -1))
    return binary_crossentropy(y_true, y_pred)


def custom_loss_class_wrapper(n):
    def custom_loss_class(y_true, y_pred):
        # The model will be trained using this loss function, which is identical to normal BCE
        # except when the label for an example is -1, that example is masked out for that task.
        # This allows for examples to only impact loss backpropagation for one of the two tasks.
        y_pred = tensorflow.boolean_mask(y_pred, tensorflow.not_equal(y_true, [-1]*n))
        y_true = tensorflow.boolean_mask(y_true, tensorflow.not_equal(y_true, [-1]*n))
        return categorical_crossentropy(y_true, y_pred)
    return custom_loss_class

def custom_loss(y_true, y_pred):
    # The model will be trained using this loss function, which is identical to normal BCE
    # except when the label for an example is -1, that example is masked out for that task.
    # This allows for examples to only impact loss backpropagation for one of the two tasks.
    y_pred = tensorflow.boolean_mask(y_pred, tensorflow.not_equal(y_true, -1))
    y_true = tensorflow.boolean_mask(y_true, tensorflow.not_equal(y_true, -1))
    return binary_crossentropy(y_true, y_pred)


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

        if model_type == 'kelly':
            model_file_prefix = ROOT + "/".join(["kelly_models", tf, train_species + "_trained", 'DA']) + "/"      

        else:
            model_file_prefix = ROOT + "/".join(["models", tf, train_species + "_trained", model_type]) + "/"      
        
        # these models were saved as part of training
        # see ../2_train_and_test_models/callbacks.py for model saving details 
        try:
            model_file_suffix = "_run" + str(run) + "_best.model"
            files = [f for f in os.listdir(model_file_prefix) if f.endswith(model_file_suffix)]
            latest_file = max([model_file_prefix + f for f in files], key=os.path.getctime)
        except:
            model_file_suffix = "_run" + str(run) + "_15E_end.model"
            files = [f for f in os.listdir(model_file_prefix) if f.endswith(model_file_suffix)]
            latest_file = max([model_file_prefix + f for f in files], key=os.path.getctime)
        
        # # get all files that match the prefix and suffix
        # files = [f for f in os.listdir(model_file_prefix) if f.endswith(model_file_suffix)]
        
        # # sort files and return the one that is most recent
        # latest_file = max([model_file_prefix + f for f in files], key=os.path.getctime)
        return latest_file


def load_keras_model(model_file, n, DA = False):
    print("Loading " + model_file + ".")
    if DA:
        # need to tell Keras how the GRL and the custom loss was implemented
        # (these need to match the definitions from when the model was saved)
        return keras.models.load_model(model_file,
                custom_objects = {"GradientReversal":GradientReversal,
                                  "custom_loss":custom_loss})

    return keras.models.load_model(model_file)


def get_models_all_runs(tf, train_species, model_type, runs = 1):
    # load in models for all runs, for a given TF and training species
    # returns a list of Keras model objects
    models = []
    for run in range(runs):
        model_file = get_model_file(tf, train_species, run + 1, model_type = model_type)
        if model_type == "kelly":
            models.append(load_keras_model(model_file, 2, DA = True))
        else:
            models.append(load_keras_model(model_file, n, DA = False))
    return models

def get_preds_file(tf, trained, test_species, model_type):
    preds_root = ROOT + "model_out/"
    os.makedirs(preds_root, exist_ok=True)
    return preds_root + tf + "_" + model_type + "_" + trained + "-trained_" + test_species + "-test.preds"

if __name__ == "__main__":
    test_species = 'hg38'

    for train_species in SPECIES:
        if train_species in SPECIES_SMALL:
            TFS = ["CEBPA", "FoxA1", "HNF4a", "HNF6"]
        else:
            TFS = ["CEBPA"]

        for tf in TFS:
            for model_type in model_types:

                print("\n===== " + tf + " " + test_species + " test, " + train_species + " trained " + model_type + "  =====\n")

                # load the independently trained models for the given tf and training species
                models = get_models_all_runs(tf=tf, train_species=train_species, model_type=model_type)
                
                # generate predictions for all 5 independent model runs on human data
                all_model_preds = np.array([get_preds_batched_fast(model, 1024, test_species=test_species) for model in models])
                
                # if we got the output of DA model, throw out species preds and keep binding preds
                # if model_type == "DA" and len(all_model_preds.shape) > 2:
                #     all_model_preds = all_model_preds[:, 0, :]
                assert len(all_model_preds.shape) == 2, all_model_preds.shape
                
                # save predictions to file
                preds_file = get_preds_file(tf, trained, test_species, model_type)
                np.save(preds_file, all_model_preds.T)

                # clear variables and model to avoid unnecessary memory usage
                del all_model_preds, models
                keras.backend.clear_session()




