import pandas as pd
import numpy as np


# Amostra Sistemática

# Importando base de dados
base = pd.read_csv('dados/Iris.csv')
base

# Criando a semente
# np.random.seed(2345)

# Criando amostra
amostra = np.random.choice(a = [0, 1], size = 150, replace = True, p = [0.5, 0.5])
amostra

len(amostra)
len(amostra[amostra == 1])
len(amostra[amostra == 0])