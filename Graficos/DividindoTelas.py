import pandas as pd
import matplotlib.pyplot as plt


# Dividindo Telas

# Carregando base de dados
base = pd.read_csv('./Dados/Trees.csv')

# Girth com Volume
plt.figure(1)
plt.subplot(2, 2, 1)
plt.scatter(base.Girth, base.Volume)

# Girth com Heigth
plt.subplot(2, 2, 2)
plt.scatter(base.Girth, base.Height)

# Heigth com Volume
plt.subplot(2, 2, 3)
plt.scatter(base.Height, base.Volume)

# Histograma Volume
plt.subplot(2, 2, 4)
plt.hist(base.Volume)
