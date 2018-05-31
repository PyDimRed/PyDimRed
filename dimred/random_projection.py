import numpy as np

class RandomProjection():


    def __init__(self, n_component):
        self.ncomponent = n_component
        self.component = None
        self.nfeatures = None



    def _fit(self, X):
        self.nfeatures = X.shape[1]
        r = np.random.mtrand._rand
        self.component = r.normal(loc = 0,
                                    scale = 1 / np.sqrt(self.ncomponent),
                                    size = (self.ncomponent, self.nfeatures ))


    def fit(self, X):
        return self._fit(X)





    def transform(self, X):

        X_new = np.dot(X, self.component)
        return X_new











