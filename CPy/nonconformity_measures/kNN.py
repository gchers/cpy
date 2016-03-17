#!/usr/local/bin/env python
import ncm_utils
import numpy as np
from scipy.spatial.distance import cdist

class KNN(ncm_utils.NCM):
    
    def __init__(self, k):
        """k-Nearest Neighbours (kNN) non-conformity measure.
        
        Keyword arguments
            k: number of neighbours
        """
        if k <= 0:
            raise Exception("Parameter `k' for k-NN should be larger than 0")
        self.k = k
        
    def compute(self, z, Z):
        """Return k-Nearest Neighbours (kNN) nonconformity measure.
        
        Keyword arguments
            z: numpy array
               The example on which to calculate the measure.
            Z: Two dimensional numpy array
               Contains examples one per row, excluding z.
        """
        # Take the k smallest distances between z rows and zn and sum them.
        dist = cdist(Z, [z])[:,0]
        r = np.sort(dist)[:self.k].sum()
    
        return r
