from sklearn.metrics import average_precision_score, roc_auc_score, confusion_matrix, log_loss
from keras.callbacks import Callback
import numpy as np
import subprocess

from generators import ValGenerator
import time


class MetricsHistory(Callback):
    # This callback calculates and saves the validation auPRCs from both species, each epoch.

    def __init__(self, parameters):
        self.auprcs = []
        # some builtin variable is called params so be careful
        self.parameters = parameters

    def on_train_begin(self, logs={}):
        params = self.parameters

        # start = time.time()

        print("Validating on target species...")
        if isinstance(params.target_species, list):
            params.target_val_probs = []
            target_auprc = []
            for species in params.target_species:
                current_probs = self.model.predict_generator(ValGenerator(params, target_species=species),
                                                                use_multiprocessing = True,
                                                                workers = 8)
                params.target_val_probs.append(current_probs)
                target_auprc.append(self.print_val_metrics(params, target_data = species, source_data=None, current_probs=current_probs))
            target_auprc = sum(target_auprc)/len(target_auprc)
        else:
            params.target_val_probs = self.model.predict_generator(ValGenerator(params, target_species=params.target_species),
                                                                use_multiprocessing = True,
                                                                workers = 8)
            target_auprc = self.print_val_metrics(params, target_data = params.target_species, source_data=None, current_probs=params.target_val_probs)

        print("Validating on source species...")
        if isinstance(params.source_species, list):
            params.source_val_probs = []
            source_auprc = []
            for species in params.source_species:
                current_probs = self.model.predict_generator(ValGenerator(params, source_species=species),
                                                                use_multiprocessing = True,
                                                                workers = 8)
                params.source_val_probs.append(current_probs)
                source_auprc.append(self.print_val_metrics(params, target_data = None, source_data=species, current_probs=current_probs))
            source_auprc = sum(source_auprc)/len(source_auprc)
        else:
            params.source_val_probs = self.model.predict_generator(ValGenerator(params, source_species=params.source_species),
                                                                use_multiprocessing = True,
                                                                workers = 8)
            source_auprc = self.print_val_metrics(params, target_data = None, source_data = params.source_species, current_probs=params.source_val_probs)
        
        current_auprcs = self.auprcs
        current_auprcs.append(source_auprc)
        self.auprcs = current_auprcs

        # end = time.time()
        # t = end-start
        # with open('DA_time_callbacks.txt', "a") as f:
        #     f.write(f"on train begin took {t} time, ended at {end} \n")


    def on_epoch_end(self, batch, logs={}):
        params = self.parameters

        current_auprcs = self.auprcs
        # start = time.time()
        
        print("Validating on target species...")
        if isinstance(params.target_species, list):
            params.target_val_probs = []
            target_auprc = []
            for species in params.target_species:
                current_probs = self.model.predict_generator(ValGenerator(params, target_species=species),
                                                                use_multiprocessing = True,
                                                                workers = 8)
                params.target_val_probs.append(current_probs)
                target_auprc.append(self.print_val_metrics(params, target_data = species, source_data=None, current_probs=current_probs))
        else:
            params.target_val_probs = self.model.predict_generator(ValGenerator(params, target_species=params.target_species),
                                                                use_multiprocessing = True,
                                                                workers = 8)
            target_auprc = self.print_val_metrics(params, target_data = params.target_species, source_data=None, current_probs=params.target_val_probs)
            
            if len(current_auprcs) == 0 or target_auprc > max(current_auprcs):
                print("Best model so far! (target species) validation auPRC = ", target_auprc)
                self.model.save(params.modelfile + "_best.model")
            current_auprcs.append(target_auprc)
            self.auprcs = current_auprcs

        print("Validating on source species...")
        if isinstance(params.source_species, list):
            params.source_val_probs = []
            source_auprc = []
            for species in params.source_species:
                current_probs = self.model.predict_generator(ValGenerator(params, source_species=species),
                                                                use_multiprocessing = True,
                                                                workers = 8)
                params.source_val_probs.append(current_probs)
                source_auprc.append(self.print_val_metrics(params, target_data = None, source_data=species, current_probs=current_probs))
        else:
            params.source_val_probs = self.model.predict_generator(ValGenerator(params, source_species=params.source_species),
                                                                use_multiprocessing = True,
                                                                workers = 8)
            source_auprc = self.print_val_metrics(params, target_data = None, source_data = params.source_species, current_probs=params.source_val_probs)
            if len(current_auprcs) == 0 or source_auprc > max(current_auprcs):
                print("Best model so far! (source species) validation auPRC = ", source_auprc)
                self.model.save(params.modelfile + "_best.model")
            current_auprcs.append(source_auprc)
            self.auprcs = current_auprcs

        # end = time.time()
        # t = end-start
        # with open('DA_time_callbacks.txt', "a") as f:
        #     f.write(f"Callbacks on epoch end took {t} time, ended at {end}\n")

    def print_val_metrics(self, params, epoch_end = True, target_data = None, source_data = None, current_probs=None):
        
        # start = time.time()

        if len(current_probs) == 2: # if DA model
            current_probs = current_probs[0] # use only binding classifier preds

        if target_data:
            print("\n==== Target Species Validation: %s ====" % target_data)
            if isinstance(params.target_species, list):
                ind = params.target_species.index(target_data)
                labels = np.array(params.target_val_labels[ind])
            else:
                labels = np.array(params.target_val_labels)
            probs = np.array(current_probs)
        elif source_data:
            print("\n==== Source Species Validation: %s ====" % source_data)
            if isinstance(params.source_species, list):
                ind = params.source_species.index(source_data)
                labels = np.array(params.source_val_labels[ind])
            else:
                labels = np.array(params.source_val_labels)
            probs = np.array(current_probs)

        probs = probs.squeeze()
        assert labels.shape == probs.shape, (labels.shape, probs.shape)

        print("AUC:\t", roc_auc_score(labels, probs))
        auPRC = average_precision_score(labels, probs)
        print("auPRC:\t", auPRC)
        loss = log_loss(labels, probs)
        print("Loss:\t", loss)
        self.print_confusion_matrix(labels, probs)

        return auPRC

        # end = time.time()
        # t = end-start
        # with open('DA_time_callbacks.txt', "a") as f:
        #     f.write(f"print val metrics took {t} time, ended at {end}\n")

    def print_confusion_matrix(self, y, probs, threshold = 0.5):
        npthresh = np.vectorize(lambda t: 1 if t >= threshold else 0)
        pred = npthresh(probs)
        conf_matrix = confusion_matrix(y, pred)
        print("Confusion Matrix at t = 0.5:\n", conf_matrix)
        try:
            print("Precision at t = 0.5:\t", conf_matrix[1][1] / (conf_matrix[1][1] + conf_matrix[0][1]))
        except: 
            print("Precision at t = 0.5:\t Undefined \n")
        try:
            print("Recall at t = 0.5:\t", conf_matrix[1][1] / (conf_matrix[1][1] + conf_matrix[1][0]), "\n")
        except:
            print("Recall at t = 0.5:\t Undefined \n")




class ModelSaveCallback(Callback):
    # This callback saves the model in its current state at the beginning of each epoch,
    # and at the end of training.

    def __init__(self, parameters):
        self.model_save_file = parameters.modelfile
        self.epoch_count = 0

    def on_epoch_begin(self, batch, logs={}):
        filename = self.model_save_file + "_" + str(self.epoch_count) + "E.model"
        self.model.save(filename)
        self.epoch_count += 1

    def on_train_end(self, logs={}):
        filename = self.model_save_file + "_" + str(self.epoch_count) + "E_end.model"
        self.model.save(filename)

