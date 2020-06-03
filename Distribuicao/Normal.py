from scipy import stats
from scipy.stats import norm
import matplotlib.pyplot as plt


# Distribuição Normal

# Uma cesta de objetos, a média é 8 o desvio padrão é 2, qual a probabilidade de de tirar um objeto < 6 quilos
# cdf para valores menores < que
norm.cdf(6, 8, 2)

# sf para valores maior > que
norm.sf(6, 8, 2)
1 - norm.cdf(6, 8, 2)

# Qual a probabilidade de tirar um objeto < 6 e > 10
norm.cdf(6, 8, 2) + norm.sf(10, 8, 2)

# Qual a probabilidade de tirar um objeto < 10 e > 8
norm.cdf(10, 8, 2) - norm.cdf(8, 8, 2)

# Gerando uma amostra de tamnho 100 normal
dados = norm.rvs(size = 100)
stats.probplot(dados, plot = plt)

# Realizando o teste de Shapiro
stats.shapiro(dados)
