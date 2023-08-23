
from keras.models import Sequential
from keras.layers import Dense
import matplotlib.pyplot as plt

class RedeNeural:

    ## entrada
    def __init__(self, tamanho_entrada: int, tamanho_saida: int) -> None:
        """
        Parameters:
        tamanho_entrada (int): tamanho da entrada da rede neural.
        
        tamanho_saida (int): tamanho da saída da rede neural 
        """

        ## TODO: aprimorar a rede para que a saida seja um array de inteiros
        self.modelo = Sequential()
        self.modelo.add(Dense(units=100, activation='relu', input_dim=tamanho_entrada))
        
        self.modelo.add(Dense(units=10, activation='relu'))
        self.modelo.add(Dense(units=tamanho_saida, activation='sigmoid'))

    def compile(self):
        self.modelo.compile(loss='mse', optimizer='adam', metrics=['mae', 'accuracy'])

    def train(self, entrada, esperado, epocas=50):
        """
        Parameters:
        entrada (array bidimensional): Onde cada linha contem os pixels de uma imagem 28x28 em linha

        
        esperado (array unidimensional): Onde cada linha contem um inteiro correspondente ao caracter 
        """

        ## dividindo dados entre treino e teste
        size_teste = int(entrada.shape[0] / 4 * 3)

        entrada_treino, entrada_teste = entrada[0:size_teste], entrada[size_teste:]
        esperado_treino, esperado_teste = esperado[0:size_teste], esperado[size_teste:]

        ## compilando modelo
        self.compile()

        ## treino
        self.resultado = self.modelo.fit(
            entrada_treino, esperado_treino, 
            epochs=epocas, 
            batch_size=32, 
            validation_data=(entrada_teste, esperado_teste))
        
    def imprimir_historico_treino(self):
        plt.plot(self.resultado.history['loss'])
        plt.plot(self.resultado.history['val_loss'])
        plt.title('Histórico de Treinamento')
        plt.ylabel('Função de custo')
        plt.xlabel('Épocas de treinamento')
        plt.legend(['Erro treino', 'Erro teste'])
        plt.show()

    def imprimir_matriz_de_confusao(this, entrada, esperado):

        
        
        from sklearn.metrics import ConfusionMatrixDisplay
        from sklearn.metrics import confusion_matrix
        import matplotlib.pyplot as plt
        import numpy as np


        y_pred = np.argmax(this.modelo.predict(entrada), axis=1)
        y_test = np.argmax(esperado, axis=1)
        labels = [i for i in "0123456789"]

        cm = confusion_matrix(y_test, y_pred)

        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
        

        disp.plot(cmap=plt.cm.Blues)

        disp.ax_.set(
                title='Matriz de Confusão', 
                xlabel='Predito', 
                ylabel='Esperado')

        plt.show()

    def predict(self, valor):
        return self.modelo.predict(valor)
