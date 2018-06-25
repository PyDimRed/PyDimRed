from sklearn.neighbors import kneighbors_graph
from sklearn.neighbors import NearestNeighbors
from sklearn.utils.graph import graph_shortest_path
import numpy as np
import scipy as sp


class Laplacian():


    def __init__(self, n_component = 2, method, gamma, n_neighbours):
        self.n_neighbours = None
        self.method = None
        self.n_component = n_component
        self.gamma = None
        self.affinity = None
        self.kneargraph = None
        self.output = None


    def _fit(self, X):


        affinity = affinity_matrix(X)



    def fit(self, X):



    def affinity_matrix(self, X):
        if self.method == 'rbf':
            pass
        elif self.method == 'kneighbour':
            neighbour = NearestNeighbors(n_neighbors=self.n_neighbour)
            neighbour.fit(X)
            self.kneargraph = kneighbors_graph(neighbour, n_neighbors=self.n_neighbour, mode='distance')
            self.affinity = 1/2 * (self.kneargraph + self.kneargraph.T)

        return self.affinity



    def transform(self, X):
        pass

