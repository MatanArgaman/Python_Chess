import tensorflow as tf
import numpy as np
from tensorflow import keras
from tensorflow.keras.layers import LeakyReLU
import matplotlib.pyplot as plt
import os
import json
import argparse

from shared.shared_functionality import INPUT_PLANES, OUTPUT_PLANES
from scipy.sparse import load_npz
from predict import get_output_representation
from nn.evaluate import single_file_evaluate


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
    parser = argparse.ArgumentParser()
    parser.add_argument('--load', action='store_true', help='continue training from existing network')
    args = parser.parse_args()

    index1_order = np.random.permutation(10)
    index2_order = np.random.permutation(10)

    with open(os.path.join(os.getcwd(), 'config.json'), 'r') as f:
        config = json.load(f)

    train_writer = tf.summary.create_file_writer(os.path.join(config['train']['nn_model_path'], 'train'))
    test_writer = tf.summary.create_file_writer(os.path.join(config['train']['nn_model_path'], 'test'))


    con_train = config['train']

    if args.load:
        model = keras.models.load_model(con_train['nn_model_path'])
    else:
        model = get_model()

    index1_test = con_train['test_index1']
    index2_test = con_train['test_index2']
    step = 0
    for epoch in range(20):
        counter = 0
        for a, i in enumerate(index1_order):
            for b, j in enumerate(index2_order):
                if i == index1_test and j == index2_test:
                    continue  # leave the last file for test

                train_score = single_file_evaluate(model, [], config,10002, [0], i, j)
                test_score = single_file_evaluate(model, [], config,10002, [0], index1_test, index2_test)
                print("train:", round(train_score,3))
                print("test:", round(test_score,3))
                train_writer.flush()
                test_writer.flush()

                with train_writer.as_default():
                    tf.summary.scalar("score", train_score, step=step)
                with test_writer.as_default():
                    tf.summary.scalar("score", test_score, step=step)

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