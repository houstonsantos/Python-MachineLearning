from scipy.stats import binom


# Distribuição Binomial

# Jogar uma moeda 5 vezes, qual a probabilidade de dar cara 3 vezes
prob = binom.pmf(3, 5, 0.5)
prob

# Passar por um sinal de 4 tempos, qual a probabilidade de pegar o sinal verde nenhuma, 1, 2, 3, 4 vezes
binom.pmf(0, 4, 0.25)
binom.pmf(1, 4, 0.25)
binom.pmf(2, 4, 0.25)
binom.pmf(3, 4, 0.25)
binom.pmf(4, 4, 0.25)

# Probabilidade acumulativa
binom.cdf(4, 4, 0.25)

# Concurso com 12 questões, qual a probabilidade de acertar 7 questões, considerando que cada questão tem 4 alternativas
binom.pmf(7, 12, 0.25)
