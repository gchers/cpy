#!/usr/local/bin/env python
"""Prediction with confidence.

Implementation of prediction with confidence [1]. The implemented non-conformity
measures so far are kNN and KDE (gaussian kernel).

Ref:                                                                         
[1] Shafer, Glenn, and Vladimir Vovk. "A tutorial on conformal prediction."  
The Journal of Machine Learning Research 9 (2008): 371-421.                  
"""
__author__ = "Giovanni Cherubin <g.chers :at: gmail.com>"
import types
import numpy as np
from scipy.spatial.distance import cdist

def predict_labelled(zn, z, y, e, ncm, **ncm_args):
    """Return a prediction set for the new object zn that contains its
    true label with probability 1-e.

    Keyword arguments:
        e: significance level [0,1].
        z: numpy.array containing the examples one per row.
        y: numpy.array containing the labels one per row.
        zn: new example (vector).
        ncm: non-conformity measure. Can be in ['kde','knn'] or a function
            defined by the user accepting the examples in z (excluding the new
            example zn)and zn.
        ncm_args: arguments passed to the non-conformity measure function.
    """
    z = np.array(z)
    y = np.array(y)
    pred = []
    y_unique = np.unique(y)
    for l in y_unique:
        z_l = z[y==l,]         # Consider only examples with label l.
        if predict_unlabelled(zn, z_l, e, ncm, **ncm_args):
            pred.append(l)

    return pred

def predict_unlabelled(zn, z, e, ncm, **ncm_args):
    """Return the prediction with confidence given a certain significance level.
    Returns True if the pvalue is greater than the significance leve, False
    otherwise.

    Keyword arguments:
        e: significance level [0,1].
        z: numpy.array containing the examples one per row.
        zn: new example (vector).
        ncm: non-conformity measure. Can be in ['kde','knn'] or a function
            defined by the user accepting the examples in z (excluding the new
            example zn)and zn.
        ncm_args: arguments passed to the non-conformity measure function.
    """
    pval = calculate_pvalue(zn, z, ncm, **ncm_args)
    return (pval > e)

def calculate_pvalue(zn, z, ncm, **ncm_args):
    """Return the pvalue for new object zn given objects z.
    
    Keyword arguments:
        z: numpy.array containing the examples one per row.
        zn: new example (vector).
        ncm: non-conformity measure. Can be one in
            NON_CONFORMITY_MEASURES.keys(), or
            a function defined by the user accepting:
                (new_example, examples, [**arguments])
        ncm_args: arguments passed to the non-conformity measure function.
    """
    z = np.vstack((z, zn))
    N = z.shape[0]
    # Non-conformity measure function
    if not isinstance(ncm, types.FunctionType):
        if NON_CONFORMITY_MEASURES.has_key(ncm):
            ncm = NON_CONFORMITY_MEASURES[ncm]
        else:
            raise Exception('Undefined non-conformity function.')
    # Calculate alphas
    a = np.zeros(N)
    for i in range(N):
        z_tmp = np.delete(z, i, 0)
        zi_tmp = z[i,:]
        a[i] = ncm(zi_tmp, z_tmp, **ncm_args)
    # Calculate p-value (smoothed cp)
    t = np.random.uniform()
    pvalue = ((a > a[N-1]).sum() + (t * (a == a[N-1]).sum())) / N
    #pvalue = float((a >= a[N-1]).sum()) / N
    return pvalue
