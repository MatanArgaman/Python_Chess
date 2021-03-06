import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.layers import LeakyReLU
from tensorflow.keras.layers import Input, Conv2D, BatchNormalization
from tensorflow.keras.layers import MaxPool2D, GlobalAvgPool2D
from tensorflow.keras.layers import Add, ReLU, Dense
from tensorflow.keras import Model
import matplotlib.pyplot as plt
import os
import json
import argparse
import subprocess
from shutil import copyfile
import pickle

from shared.shared_functionality import INPUT_PLANES, OUTPUT_PLANES
from scipy.sparse import load_npz
from predict import get_output_representation
from nn.evaluate import single_file_evaluate


def conv_batchnorm_relu(x, filters, kernel_size, strides=1):
    x = Conv2D(filters=filters, kernel_size=kernel_size, strides=strides, padding='same')(x)
    x = BatchNormalization()(x)
    x = ReLU()(x)

    return x


# Identity block

def identity_block(tensor, filters):
    x = conv_batchnorm_relu(tensor, filters=filters, kernel_size=1, strides=1)
    x = conv_batchnorm_relu(x, filters=filters, kernel_size=3, strides=1)
    x = Conv2D(filters=4 * filters, kernel_size=1, strides=1)(x)
    x = BatchNormalization()(x)

    x = Add()([tensor, x])  # skip connection
    x = ReLU()(x)

    return x


# Projection block

def projection_block(tensor, filters, strides):
    # left stream
    x = conv_batchnorm_relu(tensor, filters=filters, kernel_size=1, strides=strides)
    x = conv_batchnorm_relu(x, filters=filters, kernel_size=3, strides=1)
    x = Conv2D(filters=4 * filters, kernel_size=1, strides=1)(x)
    x = BatchNormalization()(x)

    # right stream
    shortcut = Conv2D(filters=4 * filters, kernel_size=1, strides=strides, padding='same')(tensor)
    shortcut = BatchNormalization()(shortcut)

    x = Add()([shortcut, x])  # skip connection
    x = ReLU()(x)

    return x


# Resnet block

def resnet_block(x, filters, reps, strides):
    x = projection_block(x, filters, strides)
    for _ in range(reps - 1):
        x = identity_block(x, filters)

    return x


class NNModels:
    @staticmethod
    def policy1():
        model = keras.models.Sequential([
            keras.layers.Conv2D(48, (4, 4), activation='relu', input_shape=(8, 8, INPUT_PLANES), padding='same'),
            keras.layers.MaxPooling2D(pool_size=(2, 2), strides=(1, 1), padding='same'),
            keras.layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
            keras.layers.MaxPooling2D(pool_size=(2, 2), strides=(1, 1), padding='same'),
            keras.layers.Conv2D(OUTPUT_PLANES, (3, 3), activation=keras.layers.Softmax(axis=(1, 2, 3)), padding='same'),
        ])
        opt = keras.optimizers.Adam()  # learning_rate=0.01
        model.compile(optimizer=opt, loss='mean_squared_error', metrics=['accuracy'])
        return model

    @staticmethod
    def policy2():
        inputs = keras.Input(shape=(8, 8, INPUT_PLANES))
        x = layers.Conv2D(48, (4, 4), activation='relu', input_shape=(8, 8, INPUT_PLANES), padding='same')(inputs)
        x = layers.MaxPooling2D(pool_size=(2, 2), strides=(1, 1), padding='same')(x)
        x = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(x)
        x = layers.MaxPooling2D(pool_size=(2, 2), strides=(1, 1), padding='same')(x)
        x = layers.Conv2D(OUTPUT_PLANES, (3, 3), activation='softmax', padding='same')(x)
        model = keras.Model(inputs=inputs, outputs=x, name="chess_model")
        opt = keras.optimizers.Adam()  # learning_rate=0.01
        model.compile(optimizer=opt, loss='mean_squared_error', metrics=['accuracy'])
        return model

    @staticmethod
    def policy3():
        input = Input(shape=(8, 8, INPUT_PLANES))
        x = conv_batchnorm_relu(input, filters=64, kernel_size=7, strides=1)
        x = MaxPool2D(pool_size=3, strides=1, padding='same')(x)
        x = resnet_block(x, filters=64, reps=3, strides=1)
        x = resnet_block(x, filters=128, reps=4, strides=1)
        x = resnet_block(x, filters=256, reps=6, strides=1)
        x = conv_batchnorm_relu(x, filters=OUTPUT_PLANES, kernel_size=1, strides=1)
        output = keras.layers.Softmax()(x)
        model = Model(inputs=input, outputs=output)
        opt = keras.optimizers.Adam()  # learning_rate=0.01
        model.compile(optimizer=opt, loss='mean_squared_error', metrics=['accuracy'])
        return model

    @staticmethod
    def policy4():
        input = Input(shape=(8, 8, INPUT_PLANES))
        x = conv_batchnorm_relu(input, filters=64, kernel_size=7, strides=1)
        x = MaxPool2D(pool_size=3, strides=1, padding='same')(x)
        x = resnet_block(x, filters=64, reps=3, strides=1)
        x = resnet_block(x, filters=128, reps=4, strides=1)
        x = resnet_block(x, filters=256, reps=6, strides=1)

        # will be saved along with the model to enable transfer learning from this layer
        layer_name = x.name[:x.name.index('/')]

        # to be added using transfer learning
        # # x1 - value head
        # x1 = resnet_block(x, filters=32, reps=2, strides=1)
        # x1 = tf.keras.layers.Flatten()(x1)
        # x1 = tf.keras.layers.Dense(1)(x1)
        # x1 = tf.keras.activations.tanh(x1) # range is between -1 to 1

        # x2 - policy head
        x2 = resnet_block(x, filters=OUTPUT_PLANES, reps=2, strides=1)
        x2 = conv_batchnorm_relu(x2, filters=OUTPUT_PLANES, kernel_size=1, strides=1)
        x2 = tf.keras.layers.Dense(OUTPUT_PLANES)(x2)
        x2 = keras.layers.Softmax()(x2)

        model = Model(inputs=input, outputs=x2)
        opt = keras.optimizers.Adam()  # learning_rate=0.01
        model.compile(optimizer=opt, loss='mean_squared_error', metrics=['accuracy'])
        return model, layer_name

    @staticmethod
    def get_model(model_name):
        return getattr(NNModels, model_name)()


def get_model(config):
    return NNModels.get_model(config['train']['nn_model_function'])


def get_nn_io_file(index1, index2, is_input=True):
    return load_npz(os.path.join(con_train['input_output_files_path'],
                                 con_train['input_output_files_filename'] +
                                 '{0}_{1}_{2}.npz'.format(index1, index2, 'i' if is_input else 'o')))


def get_nn_win_file(index1, index2):
    path = os.path.join(con_train['input_output_files_path'],
                        con_train['input_output_files_filename'] +
                        '{0}_{1}_v.pkl'.format(index1, index2))
    with open(path, 'rb') as f:
        c = pickle.load(f)
    return c


def save_run_configuration_settings(config, model, save_model_architecture_plot=False,
                                    transfer_learning_layer_name=None):
    """
    Save all the information needed to recreate the run
    """

    # create the directory
    try:
        os.mkdir(config['train']['nn_model_path'])
    except:
        pass
    # save the git commit hash
    import subprocess
    label = subprocess.check_output(["git", "rev-parse", "HEAD"]).strip()
    with open(os.path.join(config['train']['nn_model_path'], "git.commit"), "wb") as f:
        f.write(label)

    # save the git diff
    diff = subprocess.check_output(["git", "diff"])
    with open(os.path.join(config['train']['nn_model_path'], "git.diff"), "wb") as f:
        f.write(diff)

    # save the config file
    config_path = get_config_path()
    filename = os.path.basename(config_path)
    copyfile(config_path, os.path.join(config['train']['nn_model_path'], filename))

    # save the layer name for future transfer learning
    if transfer_learning_layer_name is not None:
        with open(os.path.join(config['train']['nn_model_path'], 'transfer_layer_name.txt'), 'w') as f:
            f.write(transfer_learning_layer_name)

    # plot the model
    keras.utils.plot_model(model, os.path.join(config['train']['nn_model_path'], "model.png"), show_shapes=True)


def get_config_path(file_name = 'config.json'):
    return os.path.join(os.getcwd(), 'configs/', file_name)


def train_model(model, config, train_writer, test_writer):
    index1_test = con_train['test_index1']
    index2_test = con_train['test_index2']
    step = 0
    for epoch in range(20):
        counter = 0
        for a, i in enumerate(index1_order):
            for b, j in enumerate(index2_order):
                if i == index1_test and j == index2_test:
                    continue  # leave the a single file (1/100 of the data) for test

                k_range = np.arange(1, 20)
                train_score = single_file_evaluate(model, config, 10000, i, j, k_range)
                test_score = single_file_evaluate(model, config, 10000, index1_test, index2_test, k_range)
                print("train:", train_score[:3].round(decimals=3))
                print("test:", test_score[:3].round(decimals=3))
                train_writer.flush()
                test_writer.flush()

                with train_writer.as_default():
                    for l, k in enumerate(k_range):
                        tf.summary.scalar("score_{0}".format(k), train_score[l], step=step)
                with test_writer.as_default():
                    for l, k in enumerate(k_range):
                        tf.summary.scalar("score_{0}".format(k), test_score[l], step=step)

                x_train = get_nn_io_file(i, j, is_input=True)
                y_train = get_nn_io_file(i, j, is_input=False)
                x_train = x_train.toarray().reshape([8, 8, -1, INPUT_PLANES]).swapaxes(0, 2).swapaxes(1, 2)
                y_train = y_train.toarray().reshape([8, 8, -1, OUTPUT_PLANES]).swapaxes(0, 2).swapaxes(1, 2)
                step += x_train.shape[0]
                model.fit(x_train, y_train, epochs=1)
                model.save(con_train['nn_model_path'])
                del model
                model = keras.models.load_model(con_train['nn_model_path'])
                print("epoch:", epoch, "round:", counter, '/', '99')
                counter += 1

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default=get_config_path(), help='configuration file path')
    parser.add_argument('--load', action='store_true', help='continue training from existing network')
    args = parser.parse_args(args=[])

    index1_order = np.random.permutation(10)
    index2_order = np.random.permutation(10)

    with open(args.config, 'r') as f:
        config = json.load(f)

    train_writer = tf.summary.create_file_writer(os.path.join(config['train']['nn_model_path'], 'train'))
    test_writer = tf.summary.create_file_writer(os.path.join(config['train']['nn_model_path'], 'test'))

    con_train = config['train']

    transfer_learning_layer_name = None
    save_model_architecture_plot = False

    # get the model (load from disk or compile)
    if args.load:
        model = keras.models.load_model(con_train['nn_model_path'])
    else:
        res = get_model(config)
        if isinstance(res, tuple):
            model = res[0]
            transfer_learning_layer_name = res[1]
        else:
            model = res
            layer_name = None
            save_model_architecture_plot = True

    save_run_configuration_settings(config, model, save_model_architecture_plot, transfer_learning_layer_name)

    train_model(model, config, train_writer, test_writer)
