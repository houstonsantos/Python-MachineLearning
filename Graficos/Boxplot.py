import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


# Boxplot
# https://matplotlib.org/gallery/statistics/boxplot_demo.html

# Carregando base de dados
base = pd.read_csv('./Dados/Trees.csv')

plt.boxplot(base.Volume)
plt.title('Árvores')
plt.xlabel('Volume')

plt.boxplot(base.Volume, vert = False, showfliers = False)
plt.title('Árvores')
plt.xlabel('Volume')

plt.boxplot(base.Volume, vert = False, showfliers = True, notch = True, patch_artist = True)
plt.title('Árvores')
plt.xlabel('Volume')

plt.boxplot(base)

plt.boxplot(base.Volume, vert = False)
plt.boxplot(base.Girth, vert = False)
plt.boxplot(base.Height, vert = False)

sns.boxplot(base.Volume).set_title('Árvores')
sns.boxplot(data = base)
