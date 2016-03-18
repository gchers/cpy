#!/usr/local/bin/env python
import ncm_utils
import numpy as np
from scipy.spatial.distance import cdist

class KNN(ncm_utils.NCM):
    
    def __init__(self, k):
        """k-Nearest Neighbours (kNN) non-conformity measure.
        
        Parameters
        ----------
        k : int
            Number of neighbours.
        """
        if k <= 0:
            raise Exception("Parameter `k' for k-NN should be larger than 0")
        self.k = k
        
    def compute(self, z, Z):
        """Return k-Nearest Neighbours (kNN) nonconformity measure.
        
        Parameters
        ----------
        z : array-like, shape (n_features,)
            Test vector, where n_features is the number of features.
        Z : array-like, shape (n_samples, n_features)
            Training vectors, where n_samples is the number of samples,
            n_features is the number of features.

        Returns
        -------
        r : float
            kNN nonconformity measure on z with respect to Z.
        """
        # Take the k smallest distances between z rows and zn and sum them.
        dist = cdist(Z, [z])[:,0]
        r = np.sort(dist)[:self.k].sum()
    
        return r
