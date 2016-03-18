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
    
    
    def __init__(self, ncm, smooth=False):
        """Initialises a Conformal Prediction object.

        Parameters
        ----------
        ncm : NCM
            Non-conformity measure object.
        smooth : bool
            Use smooth CP (smooth=True) or normal CP.
        """
        self.smooth = smooth
        self.ncm = ncm

    def predict_labelled(self, z, Z, Y, e):
        """Return a prediction set for the new object z using CP.
    
        Parameters
        ----------
        z : array-like, shape (n_features,)
            Test vector, where n_features is the number of features.
        Z : array-like, shape (n_samples, n_features)
            Training vectors, where n_samples is the number of samples,
            n_features is the number of features.
        Y : array-like, shape (n_samples,)
            Training labels. Each label Y[i] correspond to the i-th
            sample of X.
        e : float in [0,1]
            Significance level.
    
        Returns
        -------
        pred : list
            Prediction set: list of predicted labels for z.
        """
        Z = np.array(Z)
        Y = np.array(Y)
        pred = []
        for y in np.unique(Y):
            Z_y = Z[Y==y,]         # Consider only examples with label l.
            if self.predict_unlabelled(z, Z_y, e):
                pred.append(y)
    
        return pred
    
    def predict_unlabelled(self, z, Z, e):
        """Predicts if z comes from the same distribution as Z, with respect
        to a significance level e.
    
        Parameters
        ----------
        z : array-like, shape (n_features,)
            Test vector, where n_features is the number of features.
        Z : array-like, shape (n_samples, n_features)
            Training vectors, where n_samples is the number of samples,
            n_features is the number of features.
        e : float in [0,1]
            Significance level.

        Returns
        -------
        pred : bool
            True, if z comes from the same distribution as Z, False otherwise.
            
        """
        pval = self.calculate_pvalue(z, Z)

        return (pval > e)
    
    def calculate_pvalue(self, z, Z):
        """Return the pvalue for a new object z given objects Z.
        
        Parameters
        ----------
        z : array-like, shape (n_features,)
            Test vector, where n_features is the number of features.
        Z : array-like, shape (n_samples, n_features)
            Training vectors, where n_samples is the number of samples,
            n_features is the number of features.
        
        Returns
        -------
        pvalue : float
            P-value: the larger its value, the more we are confident that z
            comes from the same distribution as Z.
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
