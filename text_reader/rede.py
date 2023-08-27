
from keras.models import Sequential
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense
import matplotlib.pyplot as plt


class NeuralConvolutional:

    def __init__(self, input_size: tuple, output_size: int) -> None:
        
        self.model = Sequential()

        # Fist layer: 
        # In (28 x 28) = 780 Neurons -> out 20 x (24 x 24) = 11,520 Neurons
        # Convolution -  20 feature of (5 x 5) 
        self.model.add(Conv2D(20, (5, 5), input_shape = input_size, activation = 'relu'))

        # In 20 x (24 x 24) = 11,520 Neurons -> out 20 x (12 x 12) = 2,880 Neurons
        # Pooling - (2 , 2)
        self.model.add(MaxPooling2D(pool_size = (2, 2)))


        # # Second layer: Convolution -  20 feature of (5 x 5) 
        # self.model.add(Conv2D(20, (5, 5), activation = 'relu'))
        # # Fist layer: Pooling -  2 , 2
        # self.model.add(MaxPooling2D(pool_size=(2,2)))


        # Flattening (2 Dimensions -> 1 Dimension)
        self.model.add(Flatten())


        # Saida       
        self.model.add(Dense(units = 128, activation = 'relu'))
        self.model.add(Dense(units=output_size, activation='sigmoid'))

    def compile(self):
        self.model.compile(loss = 'binary_crossentropy', optimizer='adam', metrics=['mae', 'accuracy'])

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


class Neural2InternalLayers:

    def __init__(self, input_size: int, output_size: int) -> None:
        w, h, i = input_size
        self.model = Sequential()
        self.model.add(Flatten(input_shape=(w, h)))
        self.model.add(Dense(units=100, activation='relu'))
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

    def __init__(self, input_size: tuple, output_size: int) -> None:
        w, h, i = input_size
        self.model = Sequential()
        self.model.add(Flatten(input_shape=(w, h)))
        self.model.add(Dense(units=100, activation='relu'))        
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
