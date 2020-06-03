import numpy as np
from scipy.stats import chi2_contingency


# Qui Quadrado - (Teste de Hipótese)

# Criando matrix de dados de homens e mulheres que assistem novela
novela = np.array([[19, 6], [43, 32]])

# Teste de Qui Quadrado
chi2_contingency(novela)
