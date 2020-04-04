import keras.backend as K
from keras.layers import Dense
from keras.optimizers import adam
from keras.layers import BatchNormalization
from keras.models import Sequential

class DeepRegression:
    def __init__(self):
        self.Shape = 100
        pass
    
    def Regression_model(self):
        model = Sequential()
        model.add(Dense(50, input_shape=(self.Shape)))
        pass

    def TrainModel(self):
        pass

    def TestModel(self):
        pass