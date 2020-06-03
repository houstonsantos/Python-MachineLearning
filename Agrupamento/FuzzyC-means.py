import numpy as np
import skfuzzy
from sklearn import datasets
from sklearn.metrics import confusion_matrix


# Fuzzy C-means

# Inportando base de dados
iris = datasets.load_iris()

# Executando algoritimo
r = skfuzzy.cmeans(data = iris.data.T, c = 3, m = 2, error = 0.005, maxiter = 1000, init = None)

# Probabilidades
previsoes_porcentagem = r[1]

# Verificando probabilidade dos registros 0, 1 e 2
previsoes_porcentagem[0][0]
previsoes_porcentagem[1][0]
previsoes_porcentagem[2][0]

# Verificando previsões de acordo com a maior porcentagem atribuida
previsoes = previsoes_porcentagem.argmax(axis = 0)

# Gerando matriz de confusão
confusao = confusion_matrix(iris.target, previsoes)
