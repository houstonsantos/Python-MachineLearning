import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.formula.api as sm
from sklearn.linear_model import LinearRegression
from yellowbrick.regressor import ResidualsPlot


# Regressão Linear Simples

# Importando base de dados
base = pd.read_csv('./Dados/Cars.csv')

# Removendo coluna Unnamed: 0
base = base.drop(['Unnamed: 0'], axis = 1)

# Dividindo base entre treino e teste
X = base.iloc[:, 1].values
y = base.iloc[:, 0].values

# Calculando a correlação
correlacao = np.corrcoef(X, y)

# Transformando de array para matriz
X = X.reshape(-1, 1)

# Criando modelo de treinamento e teste
modelo = LinearRegression()
modelo.fit(X, y)

# Verificando valores de interceptação, inclinação e residuais
modelo.intercept_
modelo.coef_
modelo._residues

# Verficando R2
modelo.score(X, y)

# Valores preditos
previsoes = modelo.predict(X)

# Plotando grafico de dispersão do modelo
plt.scatter(X, y)
plt.plot(X, modelo.predict(X), color = 'red')

# Testando modelo
modelo.intercept_ + modelo.coef_ * 22
modelo.predict(22)

# Usando a biblioteca yellowbrick, valores residuais
vizualizador = ResidualsPlot(modelo)
vizualizador.fit(X, y)
vizualizador.poof()


# Regressão Linear Simples

# Importando base de dados
base = pd.read_csv('./Dados/MtCars.csv')

# Removendo coluna Unnamed: 0
base = base.drop(['Unnamed: 0'], axis = 1)

# Dividindo base entre treino e teste
X = base.iloc[:, 2].values
y = base.iloc[:, 0].values

# Calculando a correlação
correlacao = np.corrcoef(X, y)

# Transformando de array para matriz
X = X.reshape(-1, 1)

# Criando modelo de treinamento e teste
modelo = LinearRegression()
modelo.fit(X, y)

# Verificando valores de interceptação, inclinação e residuais
modelo.intercept_
modelo.coef_
modelo._residues

# Verficando R2
modelo.score(X, y)

# Valores preditos
previsoes = modelo.predict(X)

# Plotando grafico de dispersão do modelo
plt.scatter(X, y)
plt.plot(X, previsoes, color = 'red')

# Testando modelo
modelo.intercept_ + modelo.coef_ * 200
modelo.predict(200)

# Usando a biblioteca yellowbrick, valores residuais
vizualizador = ResidualsPlot(modelo)
vizualizador.fit(X, y)
vizualizador.poof()

# Regressão linear simple usando biblioteca statsmodels
modelo_ajustado = sm.ols(formula = 'mpg ~ disp', data = base)
modelo_treinado = modelo_ajustado.fit()
modelo_treinado.summary()
