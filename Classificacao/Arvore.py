import pandas as pd
import graphviz
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import export_graphviz


# Ávore de decissão

# Importando base de dados
credito = pd.read_csv('./Dados/Credit.csv')

# Separando dados e classe (Feature, Target)
preditos = credito.iloc[:, 0:20].values
classe = credito.iloc[:, 20].values

# Convertendo valores categóricos em numéricos
label_enconder = LabelEncoder()
preditos[:, 0] = label_enconder.fit_transform(preditos[:, 0])
preditos[:, 2] = label_enconder.fit_transform(preditos[:, 2])
preditos[:, 3] = label_enconder.fit_transform(preditos[:, 3])
preditos[:, 5] = label_enconder.fit_transform(preditos[:, 5])
preditos[:, 6] = label_enconder.fit_transform(preditos[:, 6])
preditos[:, 8] = label_enconder.fit_transform(preditos[:, 8])
preditos[:, 9] = label_enconder.fit_transform(preditos[:, 9])
preditos[:, 11] = label_enconder.fit_transform(preditos[:, 11])
preditos[:, 13] = label_enconder.fit_transform(preditos[:, 13])
preditos[:, 14] = label_enconder.fit_transform(preditos[:, 14])
preditos[:, 16] = label_enconder.fit_transform(preditos[:, 16])
preditos[:, 18] = label_enconder.fit_transform(preditos[:, 18])
preditos[:, 19] = label_enconder.fit_transform(preditos[:, 19])

# Dividindo base entre treino e teste
X_training, X_test, y_training, y_test = train_test_split(preditos, classe, test_size = 0.3, random_state = 0)

# Criando modelo de treinamento e teste
arvore = DecisionTreeClassifier()
arvore.fit(X_training, y_training)

# Gerando arquivo da árvore de decisão - www.webgraphviz.com
export_graphviz(arvore, out_file = 'tree.dot')

# Testando modelo
previsoes = arvore.predict(X_test)

# Gerando matriz de confusão
confusao = confusion_matrix(y_test, previsoes)

# Verificando taxa de acerto e erro
taxa_acerto = accuracy_score(y_test, previsoes)
taxa_erro = 1 - taxa_acerto
