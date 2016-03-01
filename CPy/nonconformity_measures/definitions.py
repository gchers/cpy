#!/usr/local/bin/env python

NON_CONFORMITY_MEASURES = {'kde': ncm_kde,
                           'knn': ncm_knn}

class NCM:
    """Nonconformity measure.

    Every nonconformity measure should extend this class.
    """
    
    def __init__():
       
        pass
    
    def compute(zn, z):
        """Compute the nonconformity measure for the new
        object zn.
        Return a float number
        
        Keyword arguments:
            zn: the example on which to calculate the measure.
            z: numpy.array containing examples one per row, excluding zn. 
        """
        pass
