from callbacks import *
from generators import *
from train import basic_model
from ensemble_params import Ensemble_Params
import sys, os
import tensorflow as tf

from keras.models import Model
from keras.layers import Dense, Dropout, Activation, Concatenate, Input, LSTM, Conv1D, MaxPooling1D, Reshape
from keras.optimizers import Adam
import numpy as np

ROOT = "/storage/home/vza5092/group/projects/cross-species/liver-tfs/"



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

class myPreds(tf.keras.layers.Layer):
    def __init__(self, source_species, TF, run, train_steps):
        super(myPreds, self).__init__()
        self.source_species = source_species
        self.TF = TF
        self.run = run
        self.train_steps = train_steps
  
    def call(self, inputs):
        source = self.source_species[0]
        model_file = get_model_file(self.TF, source, self.run)
        

        preds = mdl.predict(inputs, steps=self.train_steps, use_multiprocessing = True, workers = 8, verbose = 1)
        all_preds = preds

        for source in self.source_species[1:]:
            model_file = get_model_file(self.TF, source, self.run)
            mdl = keras.models.load_model(model_file)
            preds = mdl.predict(inputs, steps=self.train_steps, use_multiprocessing = True, workers = 8, verbose = 1)
            all_preds = np.concatenate((all_preds, preds), axis=1)

        return all_preds


def ensemble_model(params, submodels):

    # seq_input = Input(batch_shape = (params.batchsize, params.seqlen, 4), name = 'seq')

    seq_input = Input(shape = (params.seqlen, 4, ), name = 'seq')

    # seq = myPreds(source_species=params.source_species, TF=params.tf, 
    #     run=params.run, train_steps=params.train_steps)(seq_input)
    
    submdl = submodels[0]
    all_preds = submdl(seq_input)

    for submdl in submodels[1:]:
        preds = submdl(seq_input)
        all_preds = Concatenate(axis=1)([all_preds, preds])

    result = Dense(1, activation = 'sigmoid')(all_preds)
    inputs = seq_input

    model = Model(inputs = inputs, outputs = result)
    return model

if __name__ == "__main__":
    params = Ensemble_Params(sys.argv)
    callback = MetricsHistory(params)
    save_callback =  ModelSaveCallback(params)
    
    submodels = list()

    mm10_model = keras.models.load_model(get_model_file(params.tf, "mm10", params.run))
    mm10_model.trainable = False
    mm10_model.name = 'mm10'
    submodels.append(mm10_model)

    canFam4_model = keras.models.load_model(get_model_file(params.tf, "canFam4", params.run))
    canFam4_model.trainable = False
    canFam4_model.name = 'canFam4'
    submodels.append(canFam4_model)

    rn5_model = keras.models.load_model(get_model_file(params.tf, "rn5", params.run))
    rn5_model.trainable = False
    rn5_model.name = 'rn5'
    submodels.append(rn5_model)

    rheMac10_model = keras.models.load_model(get_model_file(params.tf, "rheMac10", params.run))
    rheMac10_model.trainable = False
    rheMac10_model.name = 'rheMac10'
    submodels.append(rheMac10_model)

    if (params.tf == 'CEBPA'):
        monDom5_model = keras.models.load_model(get_model_file(params.tf, "monDom5", params.run))
        monDom5_model.trainable = False
        monDom5_model.name = 'monDom5'
        submodels.append(monDom5_model)

        galGal6_model = keras.models.load_model(get_model_file(params.tf, "galGal6", params.run))
        galGal6_model.trainable = False
        galGal6_model.name = 'galGal6'
        submodels.append(galGal6_model)

    model = ensemble_model(params, submodels)
    model.compile(loss = "binary_crossentropy", optimizer = "adam", metrics = ["accuracy"])

    print(model.summary())

    hist = model.fit_generator(epochs = params.epochs,
                   steps_per_epoch = params.train_steps,
                   generator = EnsembleTrainGenerator(params),
                   use_multiprocessing = True, workers = 8,
                   callbacks = [callback, save_callback])

