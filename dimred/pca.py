import numpy as np
import pandas as pd
import scipy as sp

class PCA():
    """

    """
    def __init__(self, n_component):
        """

        Parameters
        ----------
        n_component
        """
        self.n_component = n_component
        self.ureduce = None


    def _fit(self, X):
        sigma = np.matmul(np.transpose(X), X) / len(X)
        U, S, V = np.linalg.svd(sigma)
        self.ureduce = U[:, 0:self.n_component]

    def fit(self, X):
        self._fit(X)
        return self

    def transform(self, X):
        return np.matmul(X, self.ureduce)

    def inv_transform(self, X):
        return np.matmul(X, np.transpose(self.ureduce))

