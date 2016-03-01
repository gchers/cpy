#!/usr/local/bin/env python
"""Prediction with confidence.

Implementation of prediction with confidence [1]. The implemented non-conformity
measures so far are kNN and KDE (gaussian kernel).

Ref:                                                                         
[1] Shafer, Glenn, and Vladimir Vovk. "A tutorial on conformal prediction."  
The Journal of Machine Learning Research 9 (2008): 371-421.                  
"""
__author__ = "Giovanni Cherubin <g.chers :at: gmail.com>"

def ncm_knn(zn, z, k):
    """Return k-Nearest Neighbours (kNN) non-conformity measure.
    
    Keyword arguments
        zn: the example on which to calculate the measure
        z: numpy.array containing examples one per row, excluding zn
        k: number of neighbours
    """
    # Take the k smallest distances between z rows and zn and sum them.
    dist = cdist(z, [zn])[:,0]
    r = np.sort(dist)[:k].sum()

    return r
