from sklearn.neighbors import kneighbors_graph
from sklearn.neighbors import NearestNeighbors
from scipy.sparse.linalg import eigsh
from scipy import sparse
import numpy as np


class Laplacian():


    def __init__(self, n_neighbours =50, n_component = 2, gamma= None, method ='kneighbour'):
        self.n_neighbour = n_neighbours
        self.n_component = n_component
        self.method = method
        self.gamma = gamma
        self.affinity = None
        self.kneargraph = None
        self.output = None



    def _fit(self, X):
        self.affinity = self.affinity_matrix(X)


        #
        laplacian, dig = sparse.csgraph.laplacian(self.affinity, normed=True, return_diag=True)
        laplacian *= -1
        lambdas, eigvec = eigsh(laplacian,sigma = 1.0, k=self.n_component + 1, which='LM')
        self.output = eigvec.T[self.n_component+1 :: -1] * dig
        self.output = self.output[1:self.n_component + 1].T

        return self

    def fit(self, X):

        opj = self._fit(X)

        return self



    def affinity_matrix(self, X):
        if self.method == 'rbf':
            pass
        elif self.method == 'kneighbour':
            neighbour = NearestNeighbors(n_neighbors=self.n_neighbour)
            neighbour.fit(X)
            self.kneargraph = kneighbors_graph(neighbour, n_neighbors=self.n_neighbour, mode='distance')
            self.affinity = 1/2 * (self.kneargraph + self.kneargraph.T)

        return self.affinity



    def transform(self):
        return self.output



    def fit_transform(self, X):


        obj = self._fit(X)
        return self.output


