import igraph
from igraph import Graph
from igraph import plot


# Introdução

# Grafo direcionado
grafo1 = Graph(edges = [(0,1),(1,2),(2,3),(3,0)], directed = True)
grafo1.vs['label'] = range(grafo1.vcount())
plot(grafo1, bbox = (300, 300))
print(grafo1)

# Grafo direcionado
grafo2 = Graph(edges = [(0,1),(1,2),(2,3),(3,0),(0,3),(3,2),(2,1),(1,0)], directed = True)
grafo2.vs['label'] = range(grafo2.vcount())
plot(grafo2, bbox = (300,300))

# Grafo direcionado com auto relacionamento
grafo3 = Graph(edges = [(0,1),(1,2),(2,3),(3,0),(1,1)], directed = True)
grafo3.vs['label'] = range(grafo3.vcount())
plot(grafo3, bbox = (300,300))

# Grafo direcionado com auto relacionamento, adicionando novo vértice
grafo4 = Graph(edges = [(0,1),(1,2),(2,3),(3,0),(1,1)], directed = True)
grafo4.add_vertex(5)
grafo4.vs['label'] = range(grafo4.vcount())
plot(grafo4, bbox = (300,300))


# Grafo direcionado
grafo5 = Graph(edges = [(0,1),(2,2),(2,3),(3,0)], directed = True)
grafo5.vs['label'] = range(grafo5.vcount())
print(grafo5)

# Grafo não direcionado
grafo6 = Graph(edges = [(0,1),(2,2),(2,3),(3,0)], directed = False)
print(grafo6)

# Grafo não direcionado, adicionando novo vértices
grafo7 = Graph(directed = False)
grafo7.add_vertices(10)
grafo7.add_vertex(16)
grafo7.add_edges([(0,1),(2,2),(2,3),(3,0)])
print(grafo7)
plot(grafo7, bbox=(300,300))

# Grafo não direcionado, adicionando novo vértices e arestas
grafo8 = Graph(directed = False)
grafo8.add_vertices(5)
grafo8.add_edges([(0,1),(1,2),(2,3),(3,4),(4,0),(0,2),(2,1)])
grafo8.add_vertex(5)
grafo8.add_vertex(6)
grafo8.vs['label'] = ['A', 'B', 'C', 'D', 'E', 'F', 'G']
plot(grafo8, bbox=(300,300))


# Grafo não direcionado, adicionando novo vértices e arestas
grafo9 = Graph(directed = False)
grafo9.add_vertices(5)
grafo9.add_edges([(0,1),(1,2),(2,3),(3,4),(4,0),(0,2),(2,1)])
grafo9.add_vertex(5)
grafo9.add_vertex(6)
grafo9.vs['label'] = ['A', 'B', 'C', 'D', 'E', 'F', 'G']
grafo9.vs['name'] = ['A', 'B', 'C', 'D', 'E', 'F', 'G']

print(grafo9.get_adjacency())
grafo9.get_adjacency()[0,]
grafo9.get_adjacency()[0, 1]

# Imprimindo os vértices
for v in grafo9.vs:
    print(v)

plot(grafo9, bbox = (300, 300))

# Grafo direcionado, com pesos
grafo10 = Graph(edges = [(0, 1), (2, 3), (0, 2), (0, 3)], directed = True)
grafo10.vs['label'] = ['Fernando', 'Pedro', 'Jose', 'Antonio']
grafo10.vs['peso'] = [40, 30, 30, 25]

# Imprimindo os vértices
for v in grafo10.vs:
    print(v)

# Informação de do vértice
grafo10.vs[0]

# Imprimindo os arestas
for e in grafo10.es:
    print(e)

# Adicionando atributo de relacionamento e nomes as arestas
grafo10.es['TipoAmizade'] = ['Amigo', 'Inimigo', 'Inimigo', 'Amigo']

# Adicionando atributo peso das arestas
grafo10.es['weight'] = [1, 2, 1, 3]

# Adicionando atributo tipo
grafo10.vs['type'] = 'Humanos'

# Adicionando atributo Amizades
grafo10['name'] = 'Amizades'

# Acessando atributos
grafo10.es[0]
grafo10.es['TipoAmizade']

print(grafo10)
plot(grafo10, bbox = (300, 300))


# Impressão e visualização

# Crinado grafo
grafo11 = Graph(edges = [(0, 1), (2, 3), (0, 2), (0, 3)], directed = True)
grafo11.vs['label'] = ['Fernando', 'Pedro', 'Jose', 'Antonio']
grafo11.vs['peso'] = [40, 30, 30, 25]
grafo11.es['TipoAmizade'] = ['Amigo', 'Inimigo', 'Inimigo', 'Amigo']
grafo11.es['weight'] = [1, 2, 1, 3]

for v in grafo11.vs:
    print(v)

for e in grafo11.es:
    print(e)

grafo11.vs['cor'] = ['blue', 'red', 'yellow', 'green']

plot(grafo11, bbox = (300, 300), vertex_size = grafo11.vs['peso'],
     edge_width = grafo11.es['weight'],
     vertex_color = grafo11.vs['cor'],
     edge_curved = 0.4, vertex_shape = 'square')


# Métricas

# Importando grafo
grafo = igraph.load('Grafo.graphml')

print(grafo)
plot(grafo, bbox = (300,300))

# Verificando graus de entrada e saída dos vértice
grafo.degree(type = 'all')

# Verificando graus de saída dos vértice
grafo.degree(type = 'in')

# Verificando graus de entrada dos vértice
grafo.degree(type = 'out')

grau = grafo.degree(type = 'in')
plot(grafo, vertex_size = grau)

grafo.diameter(directed = True)

grafo.get_diameter()

grafo.neighborhood()

grafo2 = grafo
grafo.isomorphic(grafo2)


#