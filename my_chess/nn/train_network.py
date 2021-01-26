import tensorflow as tf
import numpy as np
from tensorflow import keras
import matplotlib.pyplot as plt
import os
import json

from shared.shared_functionality import INPUT_PLANES, OUTPUT_PLANES
from scipy.sparse import load_npz, csr_matrix
from predict import get_output_representation


def get_model():
    model = keras.models.Sequential([
        keras.layers.Conv2D(48, (4, 4), activation='relu', input_shape=(8, 8, INPUT_PLANES), padding='same'),
        keras.layers.MaxPooling2D(pool_size=(2, 2), strides=(1, 1), padding='same'),
        keras.layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
        keras.layers.MaxPooling2D(pool_size=(2, 2), strides=(1, 1), padding='same'),
        keras.layers.Conv2D(OUTPUT_PLANES, (3, 3), activation='softmax', padding='same'),
    ])
    opt = keras.optimizers.Adam()  # learning_rate=0.01
    model.compile(optimizer=opt, loss='mean_squared_error', metrics=['accuracy'])
    return model


def get_nn_io_file(index1, index2, is_input=True):
    return load_npz(os.path.join(con_train['input_output_files_path'],
                                 con_train['input_output_files_filename'] +
                                 '{0}_{1}_{2}.npz'.format(index1, index2, 'i' if is_input else 'o')))


if __name__ == '__main__':

    index1_order = np.random.permutation(10)
    index2_order = np.random.permutation(10)

    with open(os.path.join(os.getcwd(), 'config.json'), 'r') as f:
        config = json.load(f)

    con_train = config['train']

    model = get_model()

    for epoch in range(5):
        counter = 0
        for a, i in enumerate(index1_order):
            for b, j in enumerate(index2_order):
                if i == con_train['test_index1'] and j == con_train['test_index2']:
                    continue  # leave the last file for test
                x_train = get_nn_io_file(i, j, is_input=True)
                y_train = get_nn_io_file(i, j, is_input=False)
                x_train = x_train.toarray().reshape([8, 8, -1, INPUT_PLANES]).swapaxes(0, 2).swapaxes(1, 2)
                y_train = y_train.toarray().reshape([8, 8, -1, OUTPUT_PLANES]).swapaxes(0, 2).swapaxes(1, 2)
                model.fit(x_train, y_train, epochs=1)
                model.save(con_train['nn_model_path'])
                del model
                model = keras.models.load_model(con_train['nn_model_path'])

                print("epoch:", epoch, "round:", counter, '/', '99')
                counter += 1

    index1_test = con_train['test_index1']
    index2_test = con_train['test_index2']

    x_test = get_nn_io_file(index1_test, index2_test, is_input=True)
    y_test = get_nn_io_file(index1_test, index2_test, is_input=False)
    x_test = x_test.toarray().reshape([8, 8, -1, INPUT_PLANES]).swapaxes(0, 2).swapaxes(1, 2)
    y_test = y_test.toarray().reshape([8, 8, -1, OUTPUT_PLANES]).swapaxes(0, 2).swapaxes(1, 2)
    model.train(x_test, y_test)
