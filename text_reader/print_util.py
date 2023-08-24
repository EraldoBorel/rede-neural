from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import numpy as np

def print_confusion_matrix(model, input, expected, labels):

        y_pred = np.argmax(model.predict(input), axis=1)
        y_test = np.argmax(expected, axis=1)

        cm = confusion_matrix(y_test, y_pred)

        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)


        disp.plot(cmap=plt.cm.Blues)

        disp.ax_.set(
                title='Matriz de Confusão', 
                xlabel='Predito', 
                ylabel='Esperado')

        plt.show()


def print_train_history(history):
        plt.plot(history['loss'])
        plt.plot(history['val_loss'])
        plt.title('Histórico de Treinamento')
        plt.ylabel('Função de custo')
        plt.xlabel('Épocas de treinamento')
        plt.legend(['Erro treino', 'Erro teste'])
        plt.show()