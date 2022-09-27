from callbacks import *
from generators import *
from DA_params import *
from DA_single_params import *
from keras.models import Model
from keras.layers import Dense, Dropout, Activation, concatenate, Input, LSTM, Conv1D, MaxPooling1D, Reshape
from flipGradientTF import GradientReversal
from keras.losses import binary_crossentropy, categorical_crossentropy
import tensorflow as tf
import os, sys


def custom_loss_disc(y_true, y_pred):
    # The model will be trained using this loss function, which is identical to normal BCE
    # except when the label for an example is -1, that example is masked out for that task.
    # This allows for examples to only impact loss backpropagation for one of the two tasks.
    y_pred = tf.boolean_mask(y_pred, tf.not_equal(y_true, -1))
    y_true = tf.boolean_mask(y_true, tf.not_equal(y_true, -1))
    return binary_crossentropy(y_true, y_pred)


def custom_loss_class_wrapper(n):
    def custom_loss_class(y_true, y_pred):
        # The model will be trained using this loss function, which is identical to normal BCE
        # except when the label for an example is -1, that example is masked out for that task.
        # This allows for examples to only impact loss backpropagation for one of the two tasks.
        y_pred = tf.boolean_mask(y_pred, tf.not_equal(y_true, [-1]*n))
        y_true = tf.boolean_mask(y_true, tf.not_equal(y_true, [-1]*n))
        return categorical_crossentropy(y_true, y_pred)
    return custom_loss_class


def DA_model(params, n):
    # Here we specify the architecture of the domain-adaptive model.
    # See DA_params.py for specific parameters values used here.

    seq_input = Input(shape = (params.seqlen, 4, ), name = 'seq')
    seq = Conv1D(params.convfilters, params.filtersize, padding = "same")(seq_input)
    seq = Activation('relu')(seq)
    seq = MaxPooling1D(padding = "same", strides = params.strides, pool_size = params.pool_size)(seq)

    discriminator = LSTM(params.lstmnodes)(seq)
    discriminator = Dense(params.dl1nodes)(discriminator)
    discriminator = Activation('relu')(discriminator)
    discriminator = Dense(params.dl2nodes, activation = 'sigmoid')(discriminator)
    disc_result = Dense(1, activation = 'sigmoid', name = "discriminator")(discriminator)

    classifier = Reshape((params.get_reshape_size(), ))(seq)
    classifier = GradientReversal(params.lamb)(classifier)
    classifier = Dense(params.dl1nodes)(classifier)
    classifier = Activation('relu')(classifier)
    classifier = Dense(params.dl2nodes, activation = 'softmax')(classifier)
    class_result = Dense(n, activation = 'softmax', name = "classifier")(classifier)
    
    inputs = seq_input
    model = Model(inputs = inputs, outputs = [disc_result, class_result])
    return model


if __name__ == "__main__":
    
    if len(sys.argv) > 4:
        params = DA_Single_Params(sys.argv)
        n = 2
    else:
        params = DA_Params(sys.argv[:4])
        n = len(params.all_species)
    
    callback = MetricsHistory(params)
    save_callback = ModelSaveCallback(params)
    
    model = DA_model(params, n)

    model.compile(loss = [custom_loss_disc, custom_loss_class_wrapper(n)], loss_weights = [1, params.loss_weight], optimizer = "adam", metrics = ["accuracy"])
    
    print(model.summary())


    if len(sys.argv) > 4:
        hist = model.fit_generator(epochs = params.epochs,
                                steps_per_epoch = params.train_steps,
                                generator = DASingleTrainGenerator(params),
                                use_multiprocessing = True, workers = 8,
                                callbacks = [callback, save_callback])
    else:
        hist = model.fit_generator(epochs = params.epochs,
                                steps_per_epoch = params.train_steps,
                                generator = DATrainGenerator(params),
                                use_multiprocessing = True, workers = 8,
                                callbacks = [callback, save_callback])

