from rede import *
from data import get_mnist
from kfold import train_kfold, train_stratified_kfold
import matplotlib.pyplot as plt
from print_util import *

INPUT_SIZE = 784
OUTPUT_SIZE = 10
OUTPUT_LABELS = list("0123456789")



inputs, outputs = get_mnist()

neuralConv = NeuralConvolutional((28, 28, 1), OUTPUT_SIZE)
# neuralConv.compile()
# train_kfold(neuralConv.model, inputs, outputs)

# #print_train_history(neural2.result.history)
# print_confusion_matrix(neuralConv.model, inputs, outputs, OUTPUT_LABELS)




neural2 = Neural2InternalLayers((28, 28, 1), OUTPUT_SIZE)

neural2.compile()
train_stratified_kfold(neural2.model, inputs, outputs)

#print_train_history(neural2.result.history)
print_confusion_matrix(neural2.model, inputs, outputs, OUTPUT_LABELS)


neural1 = Neural1Layer(INPUT_SIZE, OUTPUT_SIZE)

neural1.compile()

train_kfold(neural1.model, inputs, outputs)

##print_train_history(neural1.result.history)
print_confusion_matrix(neural1.model, inputs, outputs, OUTPUT_LABELS)


while True:
    index = int(input("Enter a number (0 - 59999): "))
    img = inputs[index]
    plt.imshow(img.reshape(28, 28), cmap="Greys")

    new_var = img.reshape(1, 784)
    predicao = neural1.predict(new_var)

    #23426
    plt.title(f"Esperado: {(outputs[index]).argmax()} -> predição: {predicao.argmax()} :)")
    plt.show()