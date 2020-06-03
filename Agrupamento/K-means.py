import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.metrics import confusion_matrix
from sklearn.cluster import KMeans


# K-means

# Inportando base de dados
iris = datasets.load_iris()

# Verificando dados
unicos, quantidade = np.unique(iris.target, return_counts = True)

# Criando modelo de treinamento e teste
cluster = KMeans(n_clusters = 3)
cluster.fit(iris.data)

# Verificando resultados
centroides = cluster.cluster_centers_
previsoes = cluster.labels_

# Verificando dados previstos
unicos2, quantidade2 = np.unique(previsoes, return_counts = True)

# Gerando matriz de confusão
confusao = confusion_matrix(iris.target, previsoes)

# Gráfico de dispersão
plt.scatter(iris.data[previsoes == 0, 0], iris.data[previsoes == 0, 1], c = 'green', label = 'Setosa')
plt.scatter(iris.data[previsoes == 1, 0], iris.data[previsoes == 1, 1], c = 'red', label = 'Versicolor')
plt.scatter(iris.data[previsoes == 2, 0], iris.data[previsoes == 2, 1], c = 'blue', label = 'Virginica')
