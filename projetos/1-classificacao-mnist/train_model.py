import tensorflow as tf

#from tensorflow import keras
#from tensorflow.keras import layers

import keras
from keras import layers

# ---------------------------------------------------------------------------
# Projeto 1 — Classificação MNIST
#
# Requisitos (veja README.md desta pasta para detalhes completos):
#   1. Carregar o dataset MNIST via tf.keras.datasets.mnist
#   2. Normalizar as imagens para [0, 1] e ajustar o shape para (28, 28, 1)
#   3. Separar um conjunto de validação (ex: validation_split ou split manual)
#   4. Construir uma CNN com 3-4 blocos Conv2D + BatchNormalization + MaxPooling2D,
#      seguida de Dropout antes da camada de saída (10 classes, softmax)
#   5. Treinar com EarlyStopping monitorando a perda de validação
#   6. Exibir a acurácia de validação final no terminal
#   7. Salvar o modelo treinado como "model.h5"
# ---------------------------------------------------------------------------

# insira seu código aqui

from keras.datasets import mnist

# Carregando o mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# Formatando os dados 
x_train = x_train.reshape(x_train.shape[0], 28, 28, 1)
x_test = x_test.reshape(x_test.shape[0], 28, 28, 1)

# Transformando os valores de pixels greyscale para float
x_train = x_train.astype("float32")
x_test = x_test.astype("float32")

# Normalizando os valores (cada pixel/float tem resolução de 8 bits, ou 0-255)
x_train /= 255
x_test /= 255

## Inicialização dos hiperparâmetros (camadas e etc.) do modelo

model = keras.Sequential()
input_shape = (28, 28, 1)

model.add(layers.Conv2D(28, kernel_size=(3,3), input_shape=input_shape, activation="relu"))
model.add(layers.BatchNormalization())
model.add(layers.MaxPooling2D(pool_size=(2,2)))

model.add(layers.Conv2D(28, kernel_size=(3,3), padding="same",activation="relu"))
model.add(layers.BatchNormalization())
model.add(layers.MaxPooling2D(pool_size=(2,2)))

model.add(layers.Conv2D(28, kernel_size=(3,3), padding="same", activation="relu"))
model.add(layers.BatchNormalization())
model.add(layers.MaxPooling2D(pool_size=(2,2)))

model.add(layers.Flatten())
model.add(layers.Dense(128, activation="relu"))
model.add(layers.Dropout(rate=0.5))
model.add(layers.Dense(10, activation="softmax"))

# sumário
model.summary()

# Compilando o modelo

from keras.optimizers import Adam

optimizer = Adam(learning_rate=0.003)

model.compile(optimizer=optimizer, loss="sparse_categorical_crossentropy", metrics=["accuracy"])

# Treinando o modelo

from keras.callbacks import EarlyStopping

early_stopping = EarlyStopping(
    monitor="val_loss",
    patience=3,
    restore_best_weights=True
)

history = model.fit(
    x=x_train, y=y_train, 
    validation_split=0.1, 
    epochs=10, batch_size=32, 
    callbacks=[early_stopping]
)

score = model.evaluate(x_test, y_test)
print('\nAcurácia final:{}'.format(score[1]))

model.save("model.h5", save_format="h5")