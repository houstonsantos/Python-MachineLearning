import numpy as np
import tensorflow as tf
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from keras.models import Sequential
from keras.layers import Dense
from keras.utils import np_utils


# Multilayer Perceptron

# Carregando base de dados
base = datasets.load_iris()

# Separando dados e classe (Feature, Target)
preditos = base.data
classe = base.target

# Fazendo One-Hot Encoding (Dummy variable)
classe_dummy = np_utils.to_categorical(classe)

# Dividindo base entre treino e teste
X_training, X_test, y_training, y_test = train_test_split(preditos, classe_dummy, test_size = 0.3, random_state = 0)

# Criando modelo de treinamento e teste
modelo = Sequential()

# Adicionando ao modelo camada oculta com 5 neurônios e 4 neurônios na camada de entrada
modelo.add(Dense(units = 5, input_dim = 4))

# Adicionando ao modelo camada oculta com 4 neurônios
modelo.add(Dense(units = 4))

# Adicionando ao modelo camada de saída com 3 neurônios
modelo.add(Dense(units = 3, activation = 'softmax'))

# Resumo do modelo criado
modelo.summary()

# Compilando Rede Neural - ajuste dos pessos, cálculo de erros e resultado
modelo.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics = ['accuracy'])

# Treinando modelo
modelo.fit(X_training, y_training, epochs = 1000, validation_data = (X_test, y_test))

# Testando modelo
previsoes = modelo.predict(X_test)
previsoes = (previsoes > 0.5)

# Gerando matriz de confusão
y_test_matrix = [np.argmax(t) for t in y_test]
y_previsao_matrix = [np.argmax(t) for t in previsoes]
confusao = confusion_matrix(y_previsao_matrix, y_previsao_matrix)
