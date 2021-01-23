import tensorflow as tf
import numpy as np
from tensorflow import keras
import matplotlib.pyplot as plt
from shared.shared_functionality import INPUT_PLANES, OUTPUT_PLANES
from scipy.sparse import load_npz, csr_matrix
from predict import  get_output_representation

def get_model():
    model = keras.models.Sequential([
        keras.layers.Conv2D(48, (4, 4), activation='relu', input_shape=(8, 8, INPUT_PLANES), padding='same'),
        keras.layers.MaxPooling2D(pool_size=(2, 2),   strides=(1, 1),  padding='same'),
        keras.layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
        keras.layers.MaxPooling2D(pool_size=(2, 2),   strides=(1, 1),  padding='same'),
        keras.layers.Conv2D(OUTPUT_PLANES, (3, 3), activation='softmax', padding='same'),
    ])
    opt = keras.optimizers.Adam() # learning_rate=0.01
    model.compile(optimizer=opt, loss='mean_squared_error', metrics=['accuracy'])
    return model

if __name__ == '__main__':

    index1_order = np.random.permutation(10)
    index2_order = np.random.permutation(10)

    model =get_model()

    for epoch in range(5):
        for a, i in enumerate(index1_order):
            for b, j in enumerate(index2_order):
                if i==index1_order[-1] and j==index2_order[-1]:
                    continue # leave the last file for test
                x_train = load_npz('/home/blacknight/Downloads/estat{0}_{1}_i.npz'.format(i, j))
                y_train = load_npz('/home/blacknight/Downloads/estat{0}_{1}_o.npz'.format(i, j))
                x_train = x_train.toarray().reshape([8, 8, -1, INPUT_PLANES]).swapaxes(0, 2).swapaxes(1, 2)
                y_train = y_train.toarray().reshape([8, 8, -1, OUTPUT_PLANES]).swapaxes(0, 2).swapaxes(1, 2)
                model.fit(x_train, y_train, epochs=1)
                model.save('/home/blacknight/Downloads/mymodel')
                del model
                model = keras.models.load_model('/home/blacknight/Downloads/mymodel')

                round = a * 10 + b
                print("epoch:", epoch, "round:", round, '/','99')



    index1_test= index1_order[-1]
    index2_test= index2_order[-1]
    x_test = load_npz('/home/blacknight/Downloads/estat{0}_{1}_i.npz'.format(index1_test, index2_test))
    y_test = load_npz('/home/blacknight/Downloads/estat{0}_{1}_o.npz'.format(index1_test, index2_test))
    x_test = x_test.toarray().reshape([8, 8, -1, INPUT_PLANES]).swapaxes(0, 2).swapaxes(1,2)
    y_test = y_test.toarray().reshape([8, 8, -1, OUTPUT_PLANES]).swapaxes(0, 2).swapaxes(1,2)
    model.evaluate(x_test, y_test)



