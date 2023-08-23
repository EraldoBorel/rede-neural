from rede import RedeNeural
from data import get_mnist
from kfold import train_kfold
import matplotlib.pyplot as plt


images, labels = get_mnist()

rede_neural = RedeNeural(784, 10)

rede_neural.compile()

train_kfold(rede_neural.modelo, images, labels)

# rede_neural.train(entrada=images, esperado=labels, epocas=30)

# rede_neural.imprimir_historico_treino()

rede_neural.imprimir_matriz_de_confusao(entrada=images, esperado=labels)


while True:
    index = int(input("Enter a number (0 - 59999): "))
    img = images[index]
    plt.imshow(img.reshape(28, 28), cmap="Greys")

    new_var = img.reshape(1, 784)
    predicao = rede_neural.predict(new_var)

    #23426
    plt.title(f"Esperado: {(labels[index]).argmax()} -> predição: {predicao.argmax()} :)")
    plt.show()