import numpy as np
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.utils import np_utils
from keras.datasets import mnist
from sklearn.metrics import confusion_matrix


# Deep Learning

# Carregando base de dados e dividindo entre treino e teste
(X_training, y_training), (X_test, y_test) = mnist.load_data()

# Vizualizando imagens
plt.imshow(X_training[2], cmap = 'gray')
plt.imshow(X_training[2])
plt.title(y_training[2])

# Realizando rashape
X_training = X_training.reshape((len(X_training), np.prod(X_training.shape[1:])))
X_test = X_test.reshape((len(X_test), np.prod(X_test.shape[1:])))

# Convertendo para float
X_training = X_training.astype('float32')
X_test = X_test.astype('float32')

# Normalizando dados, colocando os valores entre 0 e 1
X_training /= 255
X_test /= 255

# Fazendo One-Hot Encoding (Dummy variable)s
y_training = np_utils.to_categorical(y_training, 10)
y_test = np_utils.to_categorical(y_test, 10)

# Criando modelo de treinamento e teste
modelo = Sequential()

# Adicionando ao modelo camada oculta com 64 neurônios e 4 neurônios na camada de entrada
modelo.add(Dense(units = 64, activation = 'relu', input_dim = 784))

# Adicionando ao modelo camada de drop-out
modelo.add(Dropout(0.2))

# Adicionando ao modelo camada oculta com 4 neurônios
modelo.add(Dense(units = 64, activation = 'relu'))
modelo.add(Dropout(0.2))

# Adicionando ao modelo camada oculta com 4 neurônios
modelo.add(Dense(units = 64, activation = 'relu'))
modelo.add(Dropout(0.2))

# Adicionando ao modelo camada de saída com 3 neurônios
modelo.add(Dense(units = 10, activation = 'softmax'))

# Resumo do modelo criado
modelo.summary()

# Compilando Rede Neural - ajuste dos pessos, cálculo de erros e resultado
modelo.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics = ['accuracy'])

# Treinando modelo
historico = modelo.fit(X_training, y_training, epochs = 20, validation_data = (X_test, y_test))

# Verificando resultados
historico.history.keys()
plt.plot(historico.history['val_loss'])
plt.plot(historico.history['val_acc'])

# Testando modelo
previsoes = modelo.predict(X_test)

y_training[20]
novo = X_training[20]
novo = np.expand_dims(novo, axis = 0)
nova_previsao = modelo.predict(novo)

# Gerando matriz de confusão
y_test_matrix = [np.argmax(t) for t in y_test]
y_previsao_matrix = [np.argmax(t) for t in previsoes]
confusao = confusion_matrix(y_previsao_matrix, y_previsao_matrix)
