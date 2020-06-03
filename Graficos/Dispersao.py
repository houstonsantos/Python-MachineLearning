import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


# Dispersão

# Carregando base de dados
base = pd.read_csv('./Dados/Trees.csv')

plt.scatter(base.Girth, base.Volume, color = 'blue', facecolors = 'none', marker = '*')
plt.title('Árvores')
plt.xlabel('Volume')
plt.ylabel('Circunferência')
plt.show()

plt.plot(base.Girth, base.Volume)

sns.regplot(base.Girth, base.Volume, data = base)
sns.regplot(base.Girth, base.Volume, data = base, x_jitter = 0.3)
sns.regplot(base.Girth, base.Volume, data = base, x_jitter = 0.3, fit_reg = False)

# Carregando base de dados
base = pd.read_csv('./Dados/CO2.csv')
x = base.conc
y = base.uptake

unicos = list(set(base.Treatment))

for i in range(len(unicos)):
    indice = base.Treatment == unicos[i]
    plt.scatter(x[indice], y[indice], label = unicos[i])
plt.legend(loc = 'lower right')

# Carregando base de dados
base = pd.read_csv('./Dados/CO2.csv')

sns.scatterplot(base.conc, base.uptake, hue = base.Type)

q = base.loc[base['Type'] == 'Quebec']
m = base.loc[base['Type'] == 'Mississippi']

plt.figure()
plt.subplot(1,2,1)
sns.scatterplot(q.conc, q.uptake).set_title('Quebec')
plt.subplot(1,2,2)
sns.scatterplot(m.conc, m.uptake).set_title('Mississippi')
plt.tight_layout()

# Refrigerado e não refrigerado
ch = base.loc[base['Treatment'] == 'chilled']
nc = base.loc[base['Treatment'] == 'nonchilled']

plt.figure()
plt.subplot(1,2,1)
sns.scatterplot(ch.conc, ch.uptake).set_title('Chilled')
plt.subplot(1,2,2)
sns.scatterplot(nc.conc, nc.uptake).set_title('Non chilled')
plt.tight_layout()

# Carregando base de dados
base = pd.read_csv('./Dados/Esoph.csv')

sns.catplot(x = 'alcgp', y = 'ncontrols', data = base, jitter = False)
sns.catplot(x = 'alcgp', y = 'ncontrols', data = base, col = 'tobgp')
