
from keras.models import Sequential
from keras.layers import Dense
import matplotlib.pyplot as plt


class Neural2InternalLayers:

    def __init__(self, input_size: int, output_size: int) -> None:
        
        self.model = Sequential()
        self.model.add(Dense(units=100, activation='relu', input_dim=input_size))
        self.model.add(Dense(units=100, activation='relu'))        
        self.model.add(Dense(units=100, activation='relu'))
        self.model.add(Dense(units=output_size, activation='sigmoid'))

    def compile(self):
        self.model.compile(loss='mse', optimizer='adam', metrics=['mae', 'accuracy'])

    def train(self, input, expected, epochs=50):
        self.compile()

        size_teste = int(input.shape[0] / 4 * 3)

        input_train, input_test = input[:size_teste], input[size_teste:]
        expected_train, expected_test = expected[:size_teste], expected[size_teste:]

        self.result = self.model.fit(
            input, expected, 
            epochs=epochs, 
            batch_size=32,
            validation_data=(input_test, expected_test))
        
    def predict(self, input):
        return self.model.predict(input)

class Neural1Layer:

    def __init__(self, input_size: int, output_size: int) -> None:
       
        self.model = Sequential()
        self.model.add(Dense(units=100, activation='relu', input_dim=input_size))        
        self.model.add(Dense(units=100, activation='relu'))
        self.model.add(Dense(units=output_size, activation='sigmoid'))

    def compile(self):
        self.model.compile(loss='mse', optimizer='adam', metrics=['mae', 'accuracy'])

    def train(self, input, expected, epochs=50):

        size_teste = int(input.shape[0] / 4 * 3)

        input_train, input_test = input[:size_teste], input[size_teste:]
        expected_train, expected_test = expected[:size_teste], expected[size_teste:]

        ## compilando modelo
        self.compile()

        ## treino
        self.result = self.model.fit(
            input, expected, 
            epochs=epochs, 
            batch_size=32, 
            validation_data=(input_test, expected_test))
        
    def predict(self, input):
        return self.model.predict(input)
