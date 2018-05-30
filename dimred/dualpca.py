import numpy as np
import pandas as pd
import scipy as sp

class DualPCA():
    """

    """
    def __init__(self, n_component):
        """

        Parameters
        ----------
        n_component
        """
        self.n_component = n_component
        self.vreduce = None
        self.sigma = None
        self.learning_X = None
    def _fit(self, X):
        self.learning_X = X
        sigma = np.matmul(np.transpose(X), X) / len(X)
        U, S, V = np.linalg.svd(sigma)
        self.sigma = S
        self.vreduce = V[:, 0:self.n_component]

    def fit(self, X):

        self._fit(X)
        return self

    def transform(self, x):
        a = np.matmul(np.transpose(self.learning_X), x)
        b = np.matmul(np.transpose(self.vreduce), a)
        c = np.matmul(np.linalg.inv(self.sigma), b)
        return c
        #return np.matmul(np.linalg.inv(self.sigma), np.matmul(np.transpose(self.vreduce),
        # np.matmul(np.transpose(self.learning_X), x)))

    def inv_transform(self, X):
        return np.matmul(X, np.matmul(self.vreduce, np.transpose(self.vreduce)))

