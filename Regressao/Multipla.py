import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.formula.api as sm
from sklearn.linear_model import LinearRegression
from yellowbrick.regressor import ResidualsPlot


# Regressão Multipla

# Importando base de dados
base = pd.read_csv('./Dados/MtCars.csv')

# Removendo coluna Unnamed: 0
base = base.drop(['Unnamed: 0'], axis = 1)

# Dividindo base entre treino e teste
X = base.iloc[:, 1:4].values
y = base.iloc[:, 0].values

# Criando modelo de treinamento e teste
modelo = LinearRegression()
modelo.fit(X, y)

# Verficando R2
modelo.score(X, y)

# Regressão linear multipla usando biblioteca statsmodels (R)
modelo_ajustado = sm.ols(formula = 'mpg ~ cyl + disp + hp', data = base)
modelo_treinado = modelo_ajustado.fit()
modelo_treinado.summary()

# Criando estrutura de dados para teste
novo = np.array([4, 200, 100])
novo = novo.reshape(1, -1)

# Testando modelo
modelo.predict(novo)
