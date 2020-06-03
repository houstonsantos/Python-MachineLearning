from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn import datasets
from scipy import stats


# Baseado em Instância

# Carregando base de dados
iris = datasets.load_iris()

# Verificando dados estatísticos da base
stats.describe(iris.data)

# Separando dados e classe (Feature, Target)
preditos = iris.data
classe = iris.target

# Dividindo base entre treino e teste
X_training, X_test, y_training, y_test = train_test_split(preditos, classe, test_size = 0.3, random_state = 0)

# Criando modelo de treinamento e teste
knn = KNeighborsClassifier(n_neighbors = 3)
knn.fit(X_training, y_training)

# Testando modelo
previsoes = knn.predict(X_test)

# Gerando matriz de confusão
confusao = confusion_matrix(y_test, previsoes)

# Verificando taxa de acerto e erro
taxa_acerto = accuracy_score(y_test, previsoes)
taxa_erro = 1 - taxa_acerto
