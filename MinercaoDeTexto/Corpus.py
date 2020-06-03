import nltk
import string
import matplotlib.pyplot as plt
from nltk.corpus import stopwords
from nltk.corpus import PlaintextCorpusReader
from matplotlib.colors import ListedColormap
from wordcloud import WordCloud
# snltk.download()


# Criando corpus
corpus = PlaintextCorpusReader('./Dados/textos', '.*')

# Colocando corpus em uma lista
arquivos = corpus.fileids()
arquivos[0]
arquivos[0:100]

# Imprimindo nomes dos arquivos
for a in arquivos:
    print(a)

# Acessando o texo de um arquivo
texto = corpus.raw('1.txt')

# Acessando todo o corpus
todo_texto = corpus.raw()

# Acessando cada palavra do corpus
palavras = corpus.words()
palavras[170]
len(palavras)

# Verificando as stop words
stops = stopwords.words('english')

# Lista de cores na nuvem de palavras
mapa_cores = ListedColormap(['orange', 'green', 'red', 'magenta'])

# Craindo nuvem de palavras
nuvem = WordCloud(background_color = 'white', colormap = mapa_cores, stopwords = stops, max_words = 100)

# Gerar a nuvem criada
nuvem.generate(todo_texto)
plt.imshow(nuvem)

# Palavras sem as stops words
palavras_semstop = [p for p in palavras if p not in stops]
len(palavras_semstop)

# Palavras sem pontuação
palavras_sem_pontuacao = [p for p in palavras_semstop if p not in string.punctuation]
len(palavras_sem_pontuacao)

# Verificando a frequência de caracteres especiais apos a retirada
frequencia = nltk.FreqDist(palavras_sem_pontuacao)

# 100 palavras com a maior frequência
mais_comuns = frequencia.most_common(100)
