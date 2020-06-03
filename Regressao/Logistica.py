import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression


# Regressão Logística

# Importando base de dados
base = pd.read_csv('./Dados/Eleicao.csv', sep =';')

# Visualizando dados da base
base.describe()

# Plotando gráfico comparativo
plt.scatter(base.DESPESAS, base.SITUACAO)

# Calculando a correlação
correlacao = np.corrcoef(base.DESPESAS, base.SITUACAO)

# Dividindo base entre treino e teste
X = base.iloc[:, 2].values
X = X[:, np.newaxis]
y = base.iloc[:, 1].values

# Criando modelo de treinamento e teste
modelo = LogisticRegression()
modelo.fit(X, y)

# Verificando valores de interceptação, inclinação
modelo.coef_
modelo.intercept_

# Plotando gráfico de previsão com sigmóide usando dados aleatórios
plt.scatter(X, y)
teste = np.linspace(10, 3000, 100)

def model(x):
    return 1 / (1 + np.exp(-x))

r = model(teste * modelo.coef_ + modelo.intercept_).ravel()
plt.plot(teste, r, color = 'red')

# Importando base de dados para teste
previsoes = pd.read_csv('dados/NovosCandidatos.csv', sep =';')

# Criando estrutura de dados para teste
despesas = previsoes.iloc[:, 1].values
despesas = despesas.reshape(-1, 1)

# Testando modelo
prev_test = modelo.predict(despesas)

# Adicionando valores preditos a base previsoes
previsoes = np.column_stack((previsoes, prev_test))
