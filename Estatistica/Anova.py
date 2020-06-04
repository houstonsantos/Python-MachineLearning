import pandas as pd
import statsmodels.api as sm
from scipy import stats
from statsmodels.formula.api import ols
from statsmodels.stats.multicomp import MultiComparison


# Anova

# Importando base de dados
tratamento = pd.read_csv('./Dados/Anova.csv', sep =';')

# Gerando gráfico box plot
tratamento.boxplot(by = 'Remedio', grid = False)

# Criando regressão para utilização da Anova
modelo1 = ols('Horas ~ Remedio', data = tratamento).fit()
resultado1 = sm.stats.anova_lm(modelo1)

# Criando regressão múltipla para utilização da Anova
modelo2 = ols('Horas ~ Remedio * Sexo', data = tratamento).fit()
resultado2 = sm.stats.anova_lm(modelo2)

# Executando teste de Tukey para comparar os resultados
mc = MultiComparison(tratamento['Horas'], tratamento['Remedio'])
resultado_teste = mc.tukeyhsd()
print(resultado_teste)

# Gerando gráfico do resultado
resultado_teste.plot_simultaneous()
