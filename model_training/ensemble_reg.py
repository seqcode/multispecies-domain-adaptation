import numpy as np
import keras
import tensorflow
import sys, os
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import LinearRegression
import joblib
from datetime import datetime
import math
import random

from default_params import *
from ensemble_params import Ensemble_Params

from flipGradientTF import GradientReversal
from keras.losses import binary_crossentropy, categorical_crossentropy

from seqdataloader.batchproducers.coordbased.core import Coordinates
from seqdataloader.batchproducers.coordbased.coordstovals.fasta import PyfaidxCoordsToVals


ROOT = "/storage/home/vza5092/group/projects/cross-species/liver-tfs/"
MODEL_ROOT = "/storage/home/vza5092/group/projects/cross-species/liver-tfs/models/"

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

if __name__ == "__main__":

    params = Ensemble_Params(sys.argv)
    timestamp = datetime.now().strftime(TIMESTAMP_FORMAT)
    params.modelfile = MODEL_ROOT + params.tf + "/" + params.target_species + "_trained/ensemble_model_reg/" + timestamp + "_run" + str(params.run)

    print(params.modelfile)

    model_type = 'kelly'

    if model_type == 'DA_single_model':
        custom_objects = {"GradientReversal":GradientReversal,
                            "custom_loss_disc":custom_loss_disc,
                            "custom_loss_class":custom_loss_class_wrapper(2)}
    elif model_type == 'kelly':
        custom_objects = {"GradientReversal":GradientReversal,
                            "custom_loss":custom_loss}
    else:
        custom_objects = {}

    submodels = list()

    mm10_model = keras.models.load_model(get_model_file(params.tf, "mm10", params.run, model_type), custom_objects=custom_objects)
    mm10_model.trainable = False
    mm10_model.name = 'mm10'
    submodels.append(mm10_model)

    canFam4_model = keras.models.load_model(get_model_file(params.tf, "canFam4", params.run, model_type), custom_objects=custom_objects)
    canFam4_model.trainable = False
    canFam4_model.name = 'canFam4'
    submodels.append(canFam4_model)

    rn5_model = keras.models.load_model(get_model_file(params.tf, "rn5", params.run, model_type), custom_objects=custom_objects)
    rn5_model.trainable = False
    rn5_model.name = 'rn5'
    submodels.append(rn5_model)

    rheMac10_model = keras.models.load_model(get_model_file(params.tf, "rheMac10", params.run, model_type), custom_objects=custom_objects)
    rheMac10_model.trainable = False
    rheMac10_model.name = 'rheMac10'
    submodels.append(rheMac10_model)

    if (params.tf == 'CEBPA'):
        monDom5_model = keras.models.load_model(get_model_file(params.tf, "monDom5", params.run, model_type), custom_objects=custom_objects)
        monDom5_model.trainable = False
        monDom5_model.name = 'monDom5'
        submodels.append(monDom5_model)

        galGal6_model = keras.models.load_model(get_model_file(params.tf, "galGal6", params.run, model_type), custom_objects=custom_objects)
        galGal6_model.trainable = False
        galGal6_model.name = 'galGal6'
        submodels.append(galGal6_model)


    # generate training data
    params.source_genome_file = "/storage/home/vza5092/group/genomes/hg38/hg38.fa"
    converter = PyfaidxCoordsToVals(params.source_genome_file)
    DATA_ROOT = ROOT + "exp_data/hg38/" + params.tf + "/"
    with open(DATA_ROOT + "chr3toY_pos_shuf.bed") as posf:
        pos_coords_tmp = [line.split()[:3] for line in posf]
        pos_coords = [Coordinates(coord[0], int(coord[1]), int(coord[2])) for coord in pos_coords_tmp]
    # tmp = 10*len(pos_coords)
    with open(DATA_ROOT + "chr3toY_neg_shuf.bed") as negf:
        # neg_coords_tmp = [next(negf).split()[:3] for x in range(tmp)]
        neg_coords_tmp = [line.split()[:3] for line in negf]
        neg_coords = [Coordinates(coord[0], int(coord[1]), int(coord[2])) for coord in neg_coords_tmp]
    random.shuffle(neg_coords)

    pos_onehot = converter(pos_coords)
    # neg_onehot = converter(neg_coords)

    factor = 1
    # factor = math.floor(len(neg_coords)/len(pos_coords))

    # all_seqs = np.concatenate((pos_onehot, neg_onehot))


    submdl = submodels[0]
    all_preds = submdl.predict(pos_onehot)[0]
    try:
        print(len(all_preds))
        print(len(all_preds[0]))
        print(all_preds)
    except:
        print("all preds is 1D")
    # print(all_preds.shape)

    for submdl in submodels[1:]:
        preds = submdl.predict(pos_onehot)[0]
        try:
            print(len(preds))
            print(len(preds[0]))
            print(preds)
        except:
            print("preds is 1D")
        all_preds = np.concatenate((all_preds, preds), axis=1)
        print(all_preds.shape)

    for i in range(factor):
        neg_onehot = converter(neg_coords[i*len(pos_coords):(i+1)*len(pos_coords)])
        submdl = submodels[0]
        neg_preds = submdl.predict(neg_onehot)[0]

        for submdl in submodels[1:]:
            preds = submdl.predict(neg_onehot)[0]
            neg_preds = np.concatenate((neg_preds, preds), axis=1)

        all_preds = np.concatenate((all_preds, neg_preds), axis=0)

    labels = np.concatenate((np.ones(pos_onehot.shape[0],), np.zeros(factor*pos_onehot.shape[0],)))

    print("Generated training data")

    regr = LinearRegression()
    regr.fit(all_preds, labels)

    print("Fit regression model")

    print("Model parameters are:")
    print(regr.get_params())
    print(regr.coef_)
    joblib.dump(regr, params.modelfile + ".joblib")
