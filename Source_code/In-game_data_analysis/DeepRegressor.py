import keras.backend as K
from keras.callbacks import Callback
from keras.callbacks import EarlyStopping
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import BatchNormalization
from keras.models import Sequential
from keras.optimizers import adam
import tensorflow as tf
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os

def get_session(gpu_fraction=0.4):
    '''Assume that you have 6GB of GPU memory and want to allocate ~2GB'''

    num_threads = os.environ.get('OMP_NUM_THREADS')
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=gpu_fraction)

    if num_threads:
        return tf.Session(config=tf.ConfigProto(
            gpu_options=gpu_options, intra_op_parallelism_threads=num_threads))
    else:
        return tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))

class PrintDot(Callback):
    def on_epoch_end(self, epoch, logs):
        if epoch % 100 == 0: print('')
        print('.', end='')

def plot_history(history):
    hist = pd.DataFrame(history.history)
    hist['epoch'] = history.epoch

    plt.figure(figsize=(8,12))

    plt.subplot(2,1,1)
    plt.xlabel('Epoch')
    plt.ylabel('Mean Abs Error [MPG]')
    plt.plot(hist['epoch'], hist['mae'],
            label='Train Error')
    plt.plot(hist['epoch'], hist['val_mae'],
            label = 'Val Error')
    plt.ylim([0,1])
    plt.legend()

    

class DeepRegression:
    def __init__(self, Shape, Data, Pred):
        self.Shape = Shape
        self.Data = Data
        self.Pred = Pred
        self.Regressor = self.Regression_model()
    
    def Regression_model(self):
        
        Adam = adam(learning_rate=0.001, beta_1=0.9, beta_2=0.999)
        
        model = Sequential()
        model.add(Dense(128, input_dim=self.Shape, activation="relu"))
        model.add(Dense(512, activation="relu"))
        model.add(BatchNormalization(momentum=0.99, epsilon=0.001))
        model.add(Dense(512, activation="relu"))
        model.add(Dropout(0.5))
        model.add(BatchNormalization(momentum=0.99, epsilon=0.001))
        model.add(Dense(512, activation="relu"))
        model.add(Dropout(0.5))
        model.add(BatchNormalization(momentum=0.99, epsilon=0.001))
        model.add(Dense(128, activation="relu"))
        model.add(Dropout(0.5))
        model.add(Dense(1, activation="linear"))
        model.compile(loss="mean_squared_error", optimizer = Adam, metrics = ["mse","mae"])

        model.summary()

        return model
        

    def Train(self, epoch):
        
        K.set_session(get_session())
        early_stop = EarlyStopping(monitor='val_loss', patience=10)
        History = self.Regressor.fit(self.Data, self.Pred, batch_size=32, epochs=epoch,validation_split = 0.2, verbose=0, callbacks=[early_stop, PrintDot()])

        hist = pd.DataFrame(History.history)
        hist['epoch'] = History.epoch
        hist.tail()

        plot_history(History)

    def Predict(self, TestData):
        Pred =  self.Regressor.predict(TestData).flatten()
        return Pred



