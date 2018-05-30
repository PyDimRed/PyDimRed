""" Dual Principal Component Analysis
"""

# Author: MohammadSadegh Akhondzadeh <ms.akhondzadeh@gmail.com>
#         Maryam Meghdadi <m.meghdadi@acm.org>

# License: BSD 3 clause
import numpy as np
import pandas as pd
import scipy as sp

class DualPCA():
    """
    Linear dimensionality reduction using Singular Value Decomposition of the
    data to project it to a lower dimensional space.

    Principal component analysis (PCA) is a statistical procedure that uses an orthogonal transformation to convert
    a set of observations of possibly correlated variables into a set of values of linearly uncorrelated variables
    called principal components

    Dual PCA is similar to PCA but it's using a trick that make this method faster in the case that the number of features
     is bigger than the samples.


    """
    def __init__(self, n_component):
        """

        Parameters
        ----------
        n_component:  int, float, None or string
           Number of components to keep.
        """
        self.n_component = n_component
        self.vreduce = None
        self.sigma = None
        self.learning_X = None
    def _fit(self, X):
        """

        Parameters
        ----------
        X : numpy.array
            Training data, shape (n_samples, n_features)

        Returns
        -------

        """
        self.learning_X = X
        sigma = np.matmul(np.transpose(X), X) / len(X)
        U, S, V = np.linalg.svd(sigma)
        self.sigma = S
        self.vreduce = V[:, 0:self.n_component]

    def fit(self, X):
        """
        Fit the model with X.
        Parameters
        ----------
        X : numpy.array
            Training data, shape (n_samples, n_features)
        Returns
        -------

        self : object
            Returns the instance itself.

        """

        self._fit(X)
        return self



    def transform(self, x):

        """
        Apply dimensionality reduction on X.

        Parameters
        ----------
        x : numpy.array
            New data, shape (n_samples_new, n_features)

        Returns
        -------
         X_new : numpy.array


        """
        a = np.matmul(np.transpose(self.learning_X), x)
        b = np.matmul(np.transpose(self.vreduce), a)
        c = np.matmul(np.linalg.inv(self.sigma), b)
        return c
        #return np.matmul(np.linalg.inv(self.sigma), np.matmul(np.transpose(self.vreduce),
        # np.matmul(np.transpose(self.learning_X), x)))

    def inv_transform(self, X):
        return np.matmul(X, np.matmul(self.vreduce, np.transpose(self.vreduce)))

