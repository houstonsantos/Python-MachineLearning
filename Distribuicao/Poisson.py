from scipy.stats import poisson


# Distribuição de Poisson

# Média de 2 acidentes por dia, probabilidade para 3 acidente
poisson.pmf(3, 2)

# Probabilidade para 3 acidente ou menos
poisson.cdf(3, 2)

# Probabilidade para mais de 3 acidente
poisson.sf(3, 2)
