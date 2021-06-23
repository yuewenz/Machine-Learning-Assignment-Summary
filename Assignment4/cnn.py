from __future__ import print_function
import tensorflow as tf
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Activation, Dropout
from tensorflow.keras.layers import LeakyReLU


class CNN(object):
    def __init__(self):
        # change these to appropriate values
        self.batch_size = 64 #number of images passed to model in each pass - a power of 2
        self.epochs = 10 #number of epochs to train over
        self.init_lr= 1e-3 #learning rate - this will be a small number

        # No need to modify these
        self.model = None

    def get_vars(self):
        return self.batch_size, self.epochs, self.init_lr

    def create_net(self):
        '''
        In this function you are going to build a convolutional neural network based on TF Keras.
        First, use Sequential() to set the inference features on this model. 
        Then, use model.add() to build layers in your own model
        Return: model
        '''

        #TODO: implement this
        self.model = Sequential()
        self.model.add(Conv2D(filters=64, kernel_size =(3, 3), strides=(1, 1), padding='same',
                              activation='relu', input_shape=(32,32,3)))
        self.model.add(Conv2D(64, (3, 3), strides=(1, 1), padding='same',activation = 'relu'))
        self.model.add(MaxPooling2D(pool_size=(2,2)))
        self.model.add(Dropout(0.25))
        self.model.add(Conv2D(128, (3, 3), strides=(1, 1), padding='same',activation='relu'))
        self.model.add(Conv2D(128, (3, 3), strides=(1, 1), padding='same',activation = 'relu'))
        self.model.add(MaxPooling2D(pool_size=(2, 2)))
        self.model.add(Conv2D(256, (3, 3), strides=(1, 1), padding='same', activation='relu'))
        self.model.add(Conv2D(256, (3, 3), strides=(1, 1), padding='same', activation='relu'))
        self.model.add(MaxPooling2D(pool_size=(2, 2)))
        # self.model.add(Conv2D(filters=64, kernel_size=(3, 3), padding='Same', activation='relu'))
        # self.model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
        # self.model.add(Dropout(0.25))
        self.model.add(Flatten())
        self.model.add(Dropout(0.5))
        self.model.add(Dense(256))
        self.model.add(Dense(10, activation='softmax'))
        return self.model

    def compile_net(self, model):
        '''
        In this function you are going to compile the model you've created.
        Use self.model.compile() to build your model.
        '''
        self.model = model

        #TODO: implement this
        self.model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

        return self.model