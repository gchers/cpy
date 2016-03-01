#!/usr/local/bin/env python
"""Prediction with confidence.

Implementation of prediction with confidence [1]. The implemented non-conformity
measures so far are kNN and KDE (gaussian kernel).

Ref:                                                                         
[1] Shafer, Glenn, and Vladimir Vovk. "A tutorial on conformal prediction."  
The Journal of Machine Learning Research 9 (2008): 371-421.                  
"""
__author__ = "Giovanni Cherubin <g.chers :at: gmail.com>"

GAUSSIAN_KERNEL = lambda u: np.exp(-0.5 * np.dot(u,u)) / np.sqrt(2 * np.pi)


class KDE(NCM):
    
    
    def __init__(h, kernel):
        """Kernel Density Estimation (KDE) non-conformity measure.
    
        Keyword arguments:
            h: bandwidth
            K: kernel function, which can be in ['gaussian',] or can be a function
                calculating the kernel over a vector and returning a scalar
        """
        self.h = h
        if kernel == 'gaussian':
             self.kernel = GAUSSIAN_KERNEL
        else:
            raise Exception('Non recognized Kernel function')
        
    def compute(zn, z):
        """Return Kernel Density Estimation (KDE) non-conformity measure.
    
        Keyword arguments:
            zn the example on which to calculate the measure
            z: numpy.array containing examples one per row, excluding zn
        """
        z_tmp = np.vstack((z, zn))
        (N, d) = z_tmp.shape
        r = - sum((self.kernel((zn - zi) / self.h) for zi in z_tmp)) / (N * (self.h ** d))
    
        return r
