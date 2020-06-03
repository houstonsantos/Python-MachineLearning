from scipy.stats import t


# Distribuição T - Student

# Média do salário dos cientistas de dados = R$ 75 reais por hora
# Amostra de tamanho 9, desvio padrão 10, z = 1.5 e 8 graus de liberdade

# Qaul a probabilidade de selecionar um cientista de dados e o salário ser menor que R$ 80 reais
t.cdf(1.5, 8)

# Qaul a probabilidade de selecionar um cientista de dados e o salário ser maior que R$ 80 reais
t.sf(1.5, 8)
