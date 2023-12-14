import baselines.GMAN.model.node2vec as node2vec
import numpy as np
import networkx as nx
from gensim.models import Word2Vec

is_directed = True
p = 2
q = 1
num_walks = 100
walk_length = 80
window_size = 10
iter = 1000
Adj_file = 'C:\\Users\\6\\PycharmProjects\\FORECASTING_GNN\\baselines\\GMAN\\adj_HOSP.txt'
SE_file = '../data/SE(hosp).txt'


def read_graph(edgelist):
    G = nx.read_edgelist(
        edgelist, nodetype=int, data=(('weight', float),),
        create_using=nx.DiGraph())

    return G


def learn_embeddings(walks, output_file):
    walks = [list(map(str, walk)) for walk in walks]
    model = Word2Vec(
        walks, window=10,vector_size=64, min_count=0, sg=1,
        workers=8)
    model.wv.save_word2vec_format(output_file)

    return


nx_G = read_graph(Adj_file)
G = node2vec.Graph(nx_G, is_directed, p, q)
G.preprocess_transition_probs()
walks = G.simulate_walks(num_walks, walk_length)
learn_embeddings(walks, SE_file)