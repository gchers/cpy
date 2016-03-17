#!/usr/local/bin/env python


class NCM:
    """Nonconformity measure.

    Every nonconformity measure should extend this class.
    """
    
    def __init__():
        """This method initialises the parameters of the
        nonconformity measure.
        """
       
    
    def compute(zn, z):
        """Compute the nonconformity measure for the new
        object zn.
        Return a float number
        
        Keyword arguments:
            zn: the example on which to calculate the measure.
            z: numpy.array containing examples one per row, excluding zn. 
        """
