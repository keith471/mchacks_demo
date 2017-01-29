from __future__ import print_function

# basic CNN model for the CIFAR-10 dataset
import sys
import numpy
from keras.datasets import cifar10
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import Flatten
from keras.constraints import maxnorm
from keras.optimizers import SGD
from keras.layers.convolutional import Convolution2D
from keras.layers.convolutional import MaxPooling2D
from keras.utils import np_utils
from keras import backend as K
K.set_image_dim_ordering('th')  # depth dimension is at index 1 instead of 3

from time import time

# for reproducibility
seed = 7
numpy.random.seed(seed)

if __name__ == '__main__':

    # load data
    t0 = time()
    # 50000 training images, 10000 testing
    # pixel values in the range 0-255
    (X_train, y_train), (X_test, y_test) = cifar10.load_data()
    print('loaded data in %fs' % (time() - t0))

    # we'll use only a fraction of the data for training since it will take too
    # long to train otherwise (about 300s per epoch)
    X_train = X_train[:10000]
    y_train = y_train[:10000]
    X_test = X_test[:2000]
    y_test = y_test[:2000]

    # we'll also want a validation set
    X_val = X_train[:2000]
    y_val = y_train[:2000]
    X_train = X_train[2000:]
    y_train = y_train[2000:]

    # normalize to 0.0-1.0
    X_train = X_train.astype('float32')
    X_test = X_test.astype('float32')
    X_val = X_val.self.astype('float32')
    X_train = X_train / 255.0
    X_test = X_test / 255.0
    X_val = X_val / 255.0

    # one hot encode outputs
    y_train = np_utils.to_categorical(y_train)
    y_test = np_utils.to_categorical(y_test)
    y_val = np.utils.to_categorical(y_val)
    num_classes = y_test.shape[1]   # 10 classes

    print('X_train shape: ', end='')
    print(X_train.shape)
    print('y_train shape: ', end='')
    print(y_train.shape)
    print('X_test shape: ', end='')
    print(X_test.shape)
    print('y_test shape: ', end='')
    print(y_test.shape)

    # define the model
    model = Sequential()
    # conv layer with 32 filters and a filter size of 3x3, relu activation
    model.add(Convolution2D(32, 3, 3, input_shape=(3, 32, 32), border_mode='same', activation='relu', W_constraint=maxnorm(3)))
    # drop 20% of the output from the conv layer
    model.add(Dropout(0.2))
    # another conv layer with 32 filters and a filter size of 3x3
    model.add(Convolution2D(32, 3, 3, activation='relu', border_mode='same', W_constraint=maxnorm(3)))
    # typical pooling layer
    model.add(MaxPooling2D(pool_size=(2, 2)))
    # flatten the output of the max pooling layer
    model.add(Flatten())
    # pass the flattened output to a dense layer with 512 outputs
    model.add(Dense(512, activation='relu', W_constraint=maxnorm(3)))
    # drop 50% of the activations
    model.add(Dropout(0.5))
    # dense output layer with softmax activation
    model.add(Dense(num_classes, activation='softmax'))

    # compile the model
    epochs = 25
    learning_rate = 0.01
    decay = learning_rate/epochs
    momentum = 0.9
    sgd = SGD(lr=learning_rate, momentum=momentum, decay=decay)
    model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])
    # print model summary

    # fit the model
    t0 = time()
    print('training...')
    model.fit(X_train, y_train, validation_data=(X_val, y_val), nb_epoch=epochs, batch_size=128)
    print('completed training in %fs' % (time() - t0))
    # evaluate against the test set
    t0 = time()
    print('testing...')
    scores = model.evaluate(X_test, y_test, verbose=0)
    print('completed testing in %fs' % (time() - t0))
    print('Accuracy: %.2f%%' % (scores[1]*100))
