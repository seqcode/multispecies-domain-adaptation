from callbacks import *
from generators import *
from default_multi_params import Multi_Params
import sys, os
import tensorflow as tf

from keras.models import Model
from keras.layers import Dense, Dropout, Activation, concatenate, Input, LSTM, Conv1D, MaxPooling1D, Reshape
from keras.optimizers import Adam
import numpy as np



def basic_model(params):
    # Here we specify the basic model architecture.
    # See default_params.py for specific values of network parameters used.

    seq_input = Input(shape = (params.seqlen, 4, ), name = 'seq')
    seq = Conv1D(params.convfilters, params.filtersize, padding = "same")(seq_input)
    seq = Activation("relu")(seq)
    seq = MaxPooling1D(padding = "same", strides = params.strides, pool_size = params.pool_size)(seq)
    
    seq = LSTM(params.lstmnodes)(seq)

    seq = Dense(params.dl1nodes, activation = "relu")(seq)
    seq = Dropout(params.dropout)(seq)
    seq = Dense(params.dl2nodes, activation = "sigmoid")(seq)
    
    result = Dense(1, activation = 'sigmoid')(seq)
    inputs = seq_input

    model = Model(inputs = inputs, outputs = result)
    return model

if __name__ == "__main__":
    params = Multi_Params(sys.argv)
    callback = MetricsHistory(params)
    save_callback =  ModelSaveCallback(params)
    
    model = basic_model(params)
    model.compile(loss = "binary_crossentropy", optimizer = "adam", metrics = ["accuracy"])

    print(model.summary())

    hist = model.fit_generator(epochs = params.epochs,
                   steps_per_epoch = params.train_steps,
                   generator = MultiTrainGenerator(params),
                   use_multiprocessing = True, workers = 8,
                   callbacks = [callback, save_callback])

