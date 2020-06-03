import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


# Barras
base = pd.read_csv('./Dados/Insect.csv')

agrupado = base.groupby(['spray'])['count'].sum()
agrupado.plot.bar(color = 'gray')

sns.countplot(base.spray)

# Pizza
agrupado.plot.pie(legend = True)
