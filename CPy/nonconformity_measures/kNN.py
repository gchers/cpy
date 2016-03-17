#!/usr/local/bin/env python
import ncm_utils

class KNN(ncm_utils.NCM):
    
    def __init__(k):
        """k-Nearest Neighbours (kNN) non-conformity measure.
        
        Keyword arguments
            k: number of neighbours
        """
        if k <= 0:
            raise Exception("Parameter `k' for k-NN should be larger than 0")
        self.k = k
        
    def compute(zn, z):
        """Return k-Nearest Neighbours (kNN) nonconformity measure.
        
        Keyword arguments
            zn: the example on which to calculate the measure
            z: numpy.array containing examples one per row, excluding zn
        """
        # Take the k smallest distances between z rows and zn and sum them.
        dist = cdist(z, [zn])[:,0]
        r = np.sort(dist)[:self.k].sum()
    
        return r
