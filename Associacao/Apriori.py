import pandas as pd
from apyori import apriori


# Apriori

# Importando base de dados
base = pd.read_csv('./Dados/Transacoes.txt', header = None)

# Modificando estrutura de dados para lista
transacoes = []
for i in range(0, 6):
    transacoes.append([str(base.values[i ,j]) for j in range(0, 3)])

# Executando algoritimo
regras = apriori(transacoes, min_support = 0.5, min_confidence = 0.5)

# Pegando resultados
resultados = list(regras)

# Visializando resultado
resultados1 = [list(x) for x in resultados]

# Visializando resultado
resultados2 = []
for j in range(0, 7):
    resultados2.append([list(x) for x in resultados1[j][2]])
