import pandas as pd
import matplotlib.pylab as plt
from datetime import datetime
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.arima_model import ARIMA
from pmdarima.arima import auto_arima


# Séries Temporais

# Importando base de dados
base = pd.read_csv('./Dados/AirPassengers.csv')

# Verificando tipo dos datos e convertendo object
print(base.dtypes)

# Criando parse para conversão
dateparse = lambda dates: pd.datetime.strptime(dates, '%Y-%m')

# Convertendo para tipo date
base = pd.read_csv('./Dados/AirPassengers.csv', parse_dates = ['Month'], index_col ='Month', date_parser = dateparse)

# Verificando os índices
base.index

# Tranformando de Data Frame para Series
ts = base['#Passengers']

# Verificando valores usando índice
ts[1]
ts['1949-02']
ts[datetime(1949,2,1)]
ts['1950-01-01':'1950-07-31']
ts[:'1950-07-31']
ts['1950']

# Verificando maior e menor valor
ts.index.max()
ts.index.min()

# Gráfico da série
plt.plot(ts)

# Agrupando por ano
ts_ano = ts.resample('A').sum()
plt.plot(ts_ano)

# Agrupando por mês
ts_mes = ts.groupby([lambda x: x.month]).sum()
plt.plot(ts_mes)

# Agrupando por ano-mês
ts_datas = ts['1960-01-01':'1960-12-01']
plt.plot(ts_datas)


# Decomposição
decomposicao = seasonal_decompose(ts)

# Valores com a tendencia de crescimento
tendencia = decomposicao.trend
plt.plot(tendencia)

# Valores sazonal
sazonal = decomposicao.seasonal
plt.plot(sazonal)

# Valores aleatórios ou residuais
aleatorio = decomposicao.resid
plt.plot(aleatorio)

# Visualizando série
plt.subplot(4, 1, 1)
plt.plot(ts, label = 'Originao')
plt.legend(loc = 'best')

plt.subplot(4, 1, 2)
plt.plot(tendencia, label = 'Tendência')
plt.legend(loc = 'best')

plt.subplot(4, 1, 3)
plt.plot(sazonal, label = 'Sazonalidade')
plt.legend(loc = 'best')

plt.subplot(4, 1, 4)
plt.plot(aleatorio, label = 'Aleatório')
plt.legend(loc = 'best')
plt.tight_layout()


# Previsão
# Verificando média
ts.mean()

# Média do ultimo ano
ts['1960-01-01':'1960-12-01'].mean()

# Média movel
# media_movel = ts.rolling(window = 2).mean()
media_movel = ts.rolling(window = 12).mean()
ts[0:12].mean()
ts[1:13].mean()

# Gráfico da série temporal
plt.plot(ts)
plt.plot(media_movel, color = 'red')

previsoes = []
for i in range(1, 13):
    superior = len(media_movel) - i
    inferior = superior - 11
    print(inferior)
    print(superior)
    print('------')
    previsoes.append(media_movel[inferior:superior].mean())

# Invertendo valores
previsoes = previsoes[:: -1]
plt.plot(previsoes)


# Previsão Usando Arima p, q, d
modelo = ARIMA(ts, order = (2, 1, 2))

# Criando modelo de treinamento e teste
modelo_treinado = modelo.fit()

# Verificando detalhes do modelo treinado
modelo_treinado.summary()

# Realizando previsão para 12 meses
previsoes = modelo_treinado.forecast(steps = 12)[0]

# Gerando gráfico de previsões
eixo = ts.plot()
modelo_treinado.plot_predict('1960-01-01', '1962-01-01', ax = eixo, plot_insample = True)

# Usando o Auto Arima para descobrir melhor conjunto de parâmetros para ARIMA
modelo_auto = auto_arima(ts, m = 12, seasonal = True, trace = True)
modelo_auto.symmary()

# Realizando previsão com Auto Arima
proximos = modelo_auto.predict(n_periods = 12)
