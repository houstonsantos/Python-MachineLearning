import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


# Histograma

# Carregando base de dados
base = pd.read_csv('./Dados/Trees.csv')

h = np.histogram(base.iloc[:, 1], bins = 'auto')
plt.hist(base.iloc[:, 1], bins = 'auto')
plt.title('Árvore')
plt.ylabel('Frequência')
plt.xlabel('Altura')

# Densidade
plt.hist(base.iloc[:, 1], bins = 10)
sns.distplot(base['Height'], color = 'blue', hist_kws = {'edgecolor': 'black'})
sns.distplot(base['Height'], color = 'blue', hist_kws = {'edgecolor': 'black'}, bins = 6)
sns.distplot(base['Height'], color = 'blue', hist_kws = {'edgecolor': 'black'}, bins = 6, hist = True, kde = True)

# Histograma condicional - comparação de uma variável contínua x categórica
sns.distplot(base.Volume, bins = 10, axlabel = 'Volume').set_title('Árvore')

# Carregando base de dados
base = pd.read_csv('./Dados/Chicken.csv')

agrupado = base.groupby(['feed'])['weight'].sum()
teste = base.loc[base['feed'] == 'horsebean']

plt.figure()
plt.subplot(3, 2, 1)
sns.distplot(base.loc[base['feed'] == 'horsebean'].weight).set_title('horsebean')

plt.subplot(3, 2, 2)
sns.distplot(base.loc[base['feed'] == 'casein'].weight).set_title('casein')

plt.subplot(3, 2, 3)
sns.distplot(base.loc[base['feed'] == 'linseed'].weight).set_title('linseed')

plt.subplot(3, 2, 4)
sns.distplot(base.loc[base['feed'] == 'meatmeal'].weight).set_title('meatmeal')

plt.subplot(3, 2, 5)
sns.distplot(base.loc[base['feed'] == 'soybean'].weight).set_title('soybean')

plt.subplot(3, 2, 6)
sns.distplot(base.loc[base['feed'] == 'sunflower'].weight).set_title('sunflower')
plt.tight_layout()
