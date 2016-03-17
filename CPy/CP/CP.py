#!/usr/local/bin/env python
"""Conformal Prediction.

Implementation of conformal predictors [1].

Ref:                                                                         
[1] Shafer, Glenn, and Vladimir Vovk. "A tutorial on conformal prediction."  
The Journal of Machine Learning Research 9 (2008): 371-421.                  
"""
__author__ = "Giovanni Cherubin <g.chers :at: gmail.com>"
import numpy as np


class CP:
    
    
    def __init__(ncm, smooth=True):
        """Initialises a Conformal Prediction object.

        Keyword arguments:
            ncm: non-conformity measure.
            smooth: use smooth CP (smooth=True) or normal CP.
        """
        self.smooth = smooth
        self.ncm = ncm

    def predict_labelled(z, Z, y, e):
        """Return a prediction set for the new object zn that contains its
        true label with probability 1-e.
    
        Keyword arguments:
            z: new example (vector).
            Z: numpy.array containing the examples one per row.
            y: numpy.array containing the labels one per row.
            e: significance level [0,1].
        """
        Z = np.array(Z)
        y = np.array(y)
        pred = []
        y_unique = np.unique(y)
        for l in y_unique:
            Z_l = Z[y==l,]         # Consider only examples with label l.
            if predict_unlabelled(z, Z_l, e):
                pred.append(l)
    
        return pred
    
    def predict_unlabelled(z, Z, e):
        """Return the prediction with confidence given a certain significance level.
        Returns True if the pvalue is greater than the significance leve, False
        otherwise.
    
        Keyword arguments:
            z: new example (vector).
            Z: numpy.array containing the examples one per row.
            e: significance level [0,1].
        """
        pval = calculate_pvalue(z, Z)

        return (pval > e)
    
    def calculate_pvalue(z, Z):
        """Return the pvalue for a new object z given objects Z.
        
        Keyword arguments:
            z: new example (vector).
            Z: numpy.array containing the examples one per row.
        """
        # Put together z and Z into Z_full.
        Z_full = np.vstack((Z, z))
        N = Z_full.shape[0]
        # Compute alphas.
        a = np.zeros(N)
        for i in range(N):
            z_tmp = Z_full[i,:]
            Z_tmp = np.delete(Z_full, i, 0)
            a[i] = self.ncm.compute(z_tmp, Z_tmp)
        # Compute p-value.
        if self.smooth == True:
            t = np.random.uniform()
            pvalue = ((a > a[N-1]).sum() + (t * (a == a[N-1]).sum())) / N
        else:
            pvalue = float((a >= a[N-1]).sum()) / N

        return pvalue
