from callbacks import *
from generators import *
from BA_params import *
from keras.models import Model
from keras.layers import Dense, Dropout, Activation, concatenate, Input, LSTM, Conv1D, MaxPooling1D, Reshape
from flipGradientTF import GradientReversal
from keras.losses import binary_crossentropy
import tensorflow as tf
import os, sys


def custom_loss(y_true, y_pred):
    # The model will be trained using this loss function, which is identical to normal BCE
    # except when the label for an example is -1, that example is masked out for that task.
    # This allows for examples to only impact loss backpropagation for one of the two tasks.
    y_pred = tf.boolean_mask(y_pred, tf.not_equal(y_true, -1))
    y_true = tf.boolean_mask(y_true, tf.not_equal(y_true, -1))
    return binary_crossentropy(y_true, y_pred)


def BA_model(params):
    # Here we specify the architecture of the domain-adaptive model.
    # See BA_params.py for specific parameters values used here.

    seq_input = Input(shape = (params.seqlen, 4, ), name = 'seq')
    seq = Conv1D(params.convfilters, params.filtersize, padding = "same")(seq_input)
    seq = Activation('relu')(seq)
    seq = MaxPooling1D(padding = "same", strides = params.strides, pool_size = params.pool_size)(seq)

    # binding
    discriminator = LSTM(params.lstmnodes)(seq)
    discriminator = Dense(params.dl1nodes)(discriminator)
    discriminator = Activation('relu')(discriminator)
    discriminator = Dense(params.dl2nodes, activation = 'sigmoid')(discriminator)
    disc_result = Dense(1, activation = 'sigmoid', name = "discriminator")(discriminator)

    # species
    classifier = Reshape((params.get_reshape_size(), ))(seq)
    classifier = GradientReversal(params.lamb)(classifier)
    classifier = Dense(params.dl1nodes)(classifier)
    classifier = Activation('relu')(classifier)
    classifier = Dense(params.dl2nodes, activation = 'sigmoid')(classifier)
    class_result = Dense(1, activation = 'sigmoid', name = "classifier")(classifier)
    
    inputs = seq_input
    model = Model(inputs = inputs, outputs = [disc_result, class_result])
    return model


if __name__ == "__main__":
    params = BA_Params(sys.argv)
    callback = MetricsHistory(params)
    save_callback = ModelSaveCallback(params)

    model = BA_model(params)

    model.compile(loss = [custom_loss, custom_loss], loss_weights = [1, params.loss_weight], optimizer = "adam", metrics = ["accuracy"])
    print(model.summary())

    hist = model.fit_generator(epochs = params.epochs,
                                steps_per_epoch = params.train_steps,
                                generator = BATrainGenerator(params),
                                use_multiprocessing = True, workers = 8,
                                callbacks = [callback, save_callback])
