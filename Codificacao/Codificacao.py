import pandas as pd
from keras.utils import np_utils
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.compose import make_column_transformer


# Categorical Encoding

# Importando base de dataset
dataset = pd.read_csv('./Dados/Credit.csv')

# Separando features para exemplo
X = dataset.iloc[:, 8:10].values

# Label Encoding
labelencoder = LabelEncoder()
X[:, 0] = labelencoder.fit_transform(X[:, 0])

# One-Hot Encoding - usando Sklearn
onehotenconder = make_column_transformer((OneHotEncoder(categories = 'auto', sparse = False), [1]), remainder = 'passthrough')
X = onehotenconder.fit_transform(X)

# One-Hot Encoding - usando Pandas
X1 = pd.get_dummies(data = dataset, columns = ['checking_status'])

# One-Hot Encoding - usando Keras - NÃO TESTADO
X2 = np_utils.to_categorical()


# Feature Scaling

# Media 0 Desvio Padrão 1
# Padronizando dataset - Z-Score
fs = dataset.iloc[:, [1, 4, 7]].values
sc = StandardScaler()
x = sc.fit_transform(fs)

# Normalizando dataset
sc = MinMaxScaler()
y = sc.fit_transform(fs)
