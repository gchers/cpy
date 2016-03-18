#!/usr/local/bin/env python


class NCM:
    """Nonconformity measure.

    Each nonconformity measure should implement this interface.
    """
    
    def __init__(self):
        """This method initialises the parameters of the
        nonconformity measure.
        """
       
    
    def compute(self, z, Z):
        """Compute the nonconformity measure for the new
        object z with respect to Z.
        
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
            Nonconformity measure on z with respect to Z.
        """
