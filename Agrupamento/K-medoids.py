import numpy as np
from sklearn import datasets
from sklearn.metrics import confusion_matrix
from pyclustering.cluster.kmedoids import kmedoids
from pyclustering.cluster import cluster_visualizer


# K-medoids

# Inportando base de dados
iris = datasets.load_iris()

# Criando modelo de treinamento e teste
cluster = kmedoids(iris.data[:, 0:2], [3, 12, 20])
cluster.get_medoids()

# Executando algoritimo
cluster.process()

# Verificando dados previstos
previsoes = cluster.get_clusters()

# Medoids, pontos centrais dos clusters
medoides = cluster.get_medoids()

# Variavel para visualização do cluster
v = cluster_visualizer()
v.append_clusters(previsoes, iris.data[:, 0:2])
v.append_cluster(medoides, data = iris.data[:, 0:2], marker = '*', markersize = 15)
v.show()

# Pegando valores preditos e reais
lista_real = []
lista_previsoes = []

for i in range(len(previsoes)):
    print('-----')
    print(i)
    print('-----')
    for j in range(len(previsoes[i])):
        print(previsoes[i][j])
        lista_previsoes.append(i)
        lista_real.append(iris.target[previsoes[i][j]])

# Convertendo estrutura de dados para confusion_matrix
lista_real = np.asanyarray(lista_real)
lista_previsoes = np.asarray(lista_previsoes)

# Gerando matriz de confusão
confusao = confusion_matrix(lista_real, lista_previsoes)
