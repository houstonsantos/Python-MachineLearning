import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import confusion_matrix, accuracy_score
from yellowbrick.classifier import ConfusionMatrix


# NavieBayes

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
# Nesta bilbioteca não é possível ver a tabela de probabilidade gerada pelo modelo
navie_bayes = GaussianNB()
navie_bayes.fit(X_training, y_training)

# Testando modelo
previsoes = navie_bayes.predict(X_test)

# Gerando matriz de confusão
confusao = confusion_matrix(y_test, previsoes)

# Verificando taxa de acerto e erro
taxa_acerto = accuracy_score(y_test, previsoes)
taxa_erro = 1 - taxa_acerto

# Gerando gráfico matriz de confusão
v = ConfusionMatrix(GaussianNB())
v.fit(X_training, y_training)
v.score(X_test, y_test)
v.show()

# Simulando Dados de Produção
novo_credito = pd.read_csv('./Dados/NovoCredit.csv')
novo_preditos = novo_credito.iloc[:, 0:20].values

# Convertendo valores categóricos em numéricos
label_enconder = LabelEncoder()
novo_preditos[:, 0] = label_enconder.fit_transform(novo_preditos[:, 0])
novo_preditos[:, 2] = label_enconder.fit_transform(novo_preditos[:, 2])
novo_preditos[:, 3] = label_enconder.fit_transform(novo_preditos[:, 3])
novo_preditos[:, 5] = label_enconder.fit_transform(novo_preditos[:, 5])
novo_preditos[:, 6] = label_enconder.fit_transform(novo_preditos[:, 6])
novo_preditos[:, 8] = label_enconder.fit_transform(novo_preditos[:, 8])
novo_preditos[:, 9] = label_enconder.fit_transform(novo_preditos[:, 9])
novo_preditos[:, 11] = label_enconder.fit_transform(novo_preditos[:, 11])
novo_preditos[:, 13] = label_enconder.fit_transform(novo_preditos[:, 13])
novo_preditos[:, 14] = label_enconder.fit_transform(novo_preditos[:, 14])
novo_preditos[:, 16] = label_enconder.fit_transform(novo_preditos[:, 16])
novo_preditos[:, 18] = label_enconder.fit_transform(novo_preditos[:, 18])
novo_preditos[:, 19] = label_enconder.fit_transform(novo_preditos[:, 19])

# Testando modelo
navie_bayes.predict(novo_preditos)
