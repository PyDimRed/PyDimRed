""" Principal Component Analysis
"""

# Author: MohammadSadegh Akhondzadeh <ms.akhondzadeh@gmail.com>
#         Maryam Meghdadi <m.meghdadi@acm.org>

# License: BSD 3 clause
import numpy as np
import pandas as pd
import scipy as sp

class PCA():
    """
    Linear dimensionality reduction using Singular Value Decomposition of the
    data to project it to a lower dimensional space.

    Principal component analysis (PCA) is a statistical procedure that uses an orthogonal transformation to convert
    a set of observations of possibly correlated variables into a set of values of linearly uncorrelated variables
    called principal components

    Here, ordinally algorithm of PCA has been implemented

    """

    def __init__(self, n_component):
        """

        Parameters
        ----------
        n_component:  int, float, None or string
           Number of components to keep.
        """
        self.n_component = n_component
        self.ureduce = None


    def _fit(self, X):
        """

        Parameters
        ----------
        X: numpy.array
            Training data

        Returns
        -------

        """
        sigma = np.matmul(np.transpose(X), X) / len(X)
        U, S, V = np.linalg.svd(sigma)
        self.ureduce = U[:, 0:self.n_component]

    def fit(self, X):
        """
        fit the model with X.

        Parameters
        ----------
        X: numpy.array
            Training data, shape (n_samples, n_features)

        Returns
        -------
        self: object
            Returns the instance itself.
        """
        self._fit(X)
        return self

    def transform(self, X):
        """
        Apply dimensionality reduction on X.

        Parameters
        ----------
        X: numpy.array
            Training data, shape (n_samples, n_features)

        Returns
        -------
        X_new: new X after applying dimensionality reduction,
                shape (n_samples, n_components)
        """
        return np.matmul(X, self.ureduce)

    def inv_transform(self, X):
        """
        Transform data back to its original space.
        Parameters
        ----------
        X: numpy.array, shape (n_samples, n_components)
            X_new

        Returns
        -------
        X_original numpy array, shape (n_samples, n_features)
        """
        return np.matmul(X, np.transpose(self.ureduce))

