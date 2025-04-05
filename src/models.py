import torch
import torch.nn as nn
import torch.nn.functional as F
import keras
from keras.preprocessing import sequence
from keras.models import Sequential
from keras.layers import Dense, Embedding
from keras.layers import Input, Flatten, Dropout, Activation
from keras.layers import Conv1D, MaxPooling1D, AveragePooling1D
from keras.models import Model

device = 0 if torch.cuda.is_available() else -1

class SEA_model():
    def __init__(self):
        self.model = Sequential()
        self.model.add(Conv1D(128, 5,padding='same',
                        input_shape=(216,1)))
        self.model.add(Activation('relu'))
        self.model.add(Conv1D(128, 5,padding='same'))
        self.model.add(Activation('relu'))
        self.model.add(Dropout(0.1))
        self.model.add(MaxPooling1D(pool_size=(8)))
        self.model.add(Conv1D(128, 5,padding='same',))
        self.model.add(Activation('relu'))
        self.model.add(Conv1D(128, 5,padding='same',))
        self.model.add(Activation('relu'))
        self.model.add(Conv1D(128, 5,padding='same',))
        self.model.add(Activation('relu'))
        self.model.add(Dropout(0.2))
        self.model.add(Conv1D(128, 5,padding='same',))
        self.model.add(Activation('relu'))
        self.model.add(Flatten())
        self.model.add(Dense(10))
        self.model.add(Activation('softmax'))
        # opt = keras.optimizers.rmsprop(lr=0.00001, decay=1e-6)

    def load_model(self):
        self.model.load_weights("../data/models/Emotion_Voice_Detection_Model.h5")
        print("Loaded model from disk")