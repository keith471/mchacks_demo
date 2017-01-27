from __future__ import print_function

# basic CNN model for the CIFAR-10 dataset
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

# load data
t0 = time()
# 50000 training images, 10000 testing
# pixel values in the range 0-255
(X_train, y_train), (X_test, y_test) = cifar10.load_data()
print('loaded data in %fs' % (time() - t0))

print()
print(X_train.shape)
print(y_train.shape)
print(X_test.shape)
print(y_test.shape)
print()
