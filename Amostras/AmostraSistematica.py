import pandas as pd
import numpy as np
from math import ceil


# Amostra Sistemática

# Dividindo base entre treino e teste
populacao = 150
amostra = 15
k = ceil(populacao / amostra)

# Realizando sorteio entre 1 e 10
r = np.random.randint(low = 1, high = k + 1, size = 1)

k
r

acumulador = r[0]
sorteados = []

for i in range(amostra):
    print(acumulador)
    # Craindo lista de sorteados
    sorteados.append(acumulador)
    acumulador += k

# Carregando arquivo iris.csv
base = pd.read_csv('dados/Iris.csv')
base_final = base.loc[sorteados]