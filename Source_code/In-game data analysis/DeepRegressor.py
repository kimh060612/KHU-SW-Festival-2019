import keras.backend as K
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import BatchNormalization
from keras.models import Sequential
from keras.optimizers import adam

class DeepRegression:
    def __init__(self, Shape, Data, Pred):
        self.Shape = Shape
        self.Data = Data
        self.Pred = Pred

        self.Regressor = self.Regression_model()
    
    def Regression_model(self):
        
        Adam = adam(learning_rate=0.001, beta_1=0.9, beta_2=0.999)
        
        model = Sequential()
        model.add(Dense(50, input_shape=(self.Shape), activation="relu", use_bias = True))
        model.add(Dense(100, activation="relu",use_bias = True))
        model.add(BatchNormalization(momentum=0.99, epsilon=0.001))
        model.add(Dense(100, activation="relu",use_bias = True))
        model.add(Dropout(0.5))
        model.add(BatchNormalization(momentum=0.99, epsilon=0.001))
        model.add(Dense(100, activation="relu",use_bias = True))
        model.add(Dropout(0.5))
        model.add(BatchNormalization(momentum=0.99, epsilon=0.001))
        model.add(Dense(10, activation="relu",use_bias = True))
        model.add(Dense(1, activation="linear"))
        model.compile(loss="mean_square_error", optimizer = Adam, metrics = ["mae"])

        model.summary()

        return model
        

    def Train(self, epoch):
        self.Regressor.fit(self.Data, self.Pred, batch_size=32, epochs=epoch)
        Loss, MAE = self.Regressor.evaluate(self.Data, self.Pred)

        


    def Predict(self):
        pass