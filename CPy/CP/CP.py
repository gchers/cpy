#!/usr/local/bin/env python
"""Conformal Prediction.

Implementation of conformal predictors [1].

Ref:                                                                         
[1] Shafer, Glenn, and Vladimir Vovk. "A tutorial on conformal prediction."  
The Journal of Machine Learning Research 9 (2008): 371-421.                  
"""
__author__ = "Giovanni Cherubin <g.chers :at: gmail.com>"
import types
import numpy as np
import non_conformity_measures
from scipy.spatial.distance import cdist


class CP:
    
    
    def __init__(ncm, smooth=True):
        """Initialises a Conformal Prediction object.

        Keyword arguments:
            ncm: non-conformity measure.
            smooth: use smooth CP (smooth=True) or normal CP.
        """
        self.smooth = smooth
        self.ncm = ncm

    def predict_labelled(zn, z, y, e):
        """Return a prediction set for the new object zn that contains its
        true label with probability 1-e.
    
        Keyword arguments:
            e: significance level [0,1].
            z: numpy.array containing the examples one per row.
            y: numpy.array containing the labels one per row.
            zn: new example (vector).
        """
        z = np.array(z)
        y = np.array(y)
        pred = []
        y_unique = np.unique(y)
        for l in y_unique:
            z_l = z[y==l,]         # Consider only examples with label l.
            if predict_unlabelled(zn, z_l, e):
                pred.append(l)
    
        return pred
    
    def predict_unlabelled(zn, z, e):
        """Return the prediction with confidence given a certain significance level.
        Returns True if the pvalue is greater than the significance leve, False
        otherwise.
    
        Keyword arguments:
            e: significance level [0,1].
            z: numpy.array containing the examples one per row.
            zn: new example (vector).
        """
        pval = calculate_pvalue(zn, z)

        return (pval > e)
    
    def calculate_pvalue(zn, z):
        """Return the pvalue for new object zn given objects z.
        
        Keyword arguments:
            z: numpy.array containing the examples one per row.
            zn: new example (vector).
        """
        z = np.vstack((z, zn))
        N = z.shape[0]
        # Compute alphas.
        a = np.zeros(N)
        for i in range(N):
            z_tmp = np.delete(z, i, 0)
            zi_tmp = z[i,:]
            a[i] = self.ncm.compute(zi_tmp, z_tmp, **ncm_args)
        # Compute p-value.
        if self.smooth:
            t = np.random.uniform()
            pvalue = ((a > a[N-1]).sum() + (t * (a == a[N-1]).sum())) / N
        else:
            pvalue = float((a >= a[N-1]).sum()) / N

        return pvalue
